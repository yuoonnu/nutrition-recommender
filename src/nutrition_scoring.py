"""
nutrition_scoring.py
영양소 정규화 + 목표별 가중치 기반 제품 스코어링 로직

사용 흐름:
1. Product 리스트로 비교 대상 제품(3~4개)을 구성한다.
2. score_products()로 목표(goal)에 맞춰 순위를 계산한다.
   - goal: 프리셋 이름을 넘기거나
   - custom_weights: 사용자가 직접/AI가 해석한 가중치 딕셔너리를 넘긴다.
3. explain_recommendation()으로 추천 근거 문구를 생성한다.
4. 자유 서술 건강정보(예: "당뇨 초기라서 혈당 관리 중이에요")는
   get_weights_from_health_text()로 가중치로 변환한 뒤 custom_weights에 넘긴다.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List

from dotenv import load_dotenv
load_dotenv()  # .env 파일을 읽어서 os.environ에 등록해줌

# google-genai 라이브러리가 출력하는 "AFC 권장" 관련 안내성 경고 로그를 숨김
# (우리는 function calling 기능을 쓰지 않으므로 무관한 메시지)
logging.getLogger("google_genai").setLevel(logging.ERROR)


@dataclass
class Product:
    name: str
    # 1회 제공량(serving) 기준 영양소 + cost. 단위는 통일해서 넣어야 함 (예: g, mg, kcal)
    # cost는 "1회 제공량당 가격"으로 환산해서 넣을 것 (패키지 가격 그대로 넣으면
    # 용량이 다른 제품끼리 비교가 왜곡됨. 예: 가격 / (패키지 총량 / 1회 제공량))
    nutrients: Dict[str, float]


# 영양소별 방향성: True면 "낮을수록 좋음", False면 "높을수록 좋음"
LOWER_IS_BETTER = {
    "calorie": True,
    "sugar": True,
    "sodium": True,
    "fat": True,
    "protein": False,
    "cost": True,
    "carb": True,
    "saturated_fat": True,
    "trans_fat": True,
    "cholesterol": True,
}

# 목표별 가중치 프리셋 (가중치는 항상 양수, 방향은 LOWER_IS_BETTER가 결정)
# 카드에 안 적힌 항목(예: 콜레스테롤 등)은 임의로 끼워넣지 않는다
GOAL_WEIGHTS = {
    # 🏋️‍♂️ 체중감량: 저칼로리 · 고단백 · 저당
    "weight_loss": {"calorie": 1.0, "protein": 0.7, "sugar": 0.6},
    # 💪 가성비 근성장: 고단백 · g당 최저가
    "muscle": {"protein": 1.0, "cost": 1.0},
    # 🩸 혈당 케어: 최저 당류 · 저탄수화물
    "blood_sugar": {"sugar": 1.2, "carb": 1.0},
    # 💧 붓기 방지: 저나트륨 · 저당 · 저지방
    "anti_swelling": {"sodium": 1.2, "sugar": 0.6, "fat": 0.6},
}

# 화면/문구용 한글 라벨
NUTRIENT_LABELS = {
    "calorie": "칼로리",
    "sugar": "당류",
    "sodium": "나트륨",
    "fat": "지방",
    "protein": "단백질",
    "cost": "가격",
    "carb": "탄수화물",
    "saturated_fat": "포화지방",
    "trans_fat": "트랜스지방",
    "cholesterol": "콜레스테롤",
}


def min_max_normalize(values: List[float]) -> List[float]:
    """0~1 사이로 정규화. 값이 전부 같으면 중립값(0.5)으로 처리."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def validate_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    사용자가 직접 설정한 가중치를 검증한다.
    - 지원하지 않는 영양소 키가 있으면 에러
    - 가중치는 0 이상이어야 함 (방향은 LOWER_IS_BETTER가 결정하므로 음수 불필요)
    - 전부 0이면 (아무것도 신경 안 쓴다는 뜻) 비교 자체가 무의미하므로 에러
    """
    for key, w in weights.items():
        if key not in LOWER_IS_BETTER:
            raise ValueError(f"지원하지 않는 항목입니다: {key}. 사용 가능: {list(LOWER_IS_BETTER)}")
        if w < 0:
            raise ValueError(f"가중치는 0 이상이어야 합니다: {key}={w}")
    if all(w == 0 for w in weights.values()):
        raise ValueError("최소 한 개 이상의 항목에 가중치를 설정해야 합니다.")
    return weights


def score_products(products: List[Product], goal: str = None, custom_weights: Dict[str, float] = None) -> List[Dict]:
    """
    제품 리스트를 목표(goal)에 맞춰 점수화하고 높은 점수 순으로 정렬해 반환.
    goal 대신 custom_weights(사용자가 직접 조정한 가중치 딕셔너리)를 넘기면
    프리셋 대신 그 값을 그대로 사용한다. (예: {"calorie": 1.0, "sodium": 0.5})
    반환 형식: [{"name": ..., "score": ..., "breakdown": {영양소: 기여도}}, ...]
    """
    if custom_weights is not None:
        weights = validate_weights(custom_weights)
    elif goal in GOAL_WEIGHTS:
        weights = GOAL_WEIGHTS[goal]
    else:
        raise ValueError(f"정의되지 않은 목표입니다: {goal}. 사용 가능: {list(GOAL_WEIGHTS)}")

    nutrient_keys = list(weights.keys())

    # 1. 영양소별로 제품군 내 min-max 정규화 후, "낮을수록 좋은" 항목은 뒤집어서
    #    항상 "값이 클수록 목표에 유리함"을 의미하도록 통일한다.
    normalized: Dict[str, List[float]] = {}
    for key in nutrient_keys:
        raw_values = [p.nutrients.get(key, 0) for p in products]
        norm_values = min_max_normalize(raw_values)
        if LOWER_IS_BETTER.get(key, True):
            norm_values = [1 - v for v in norm_values]
        normalized[key] = norm_values

    # 2. 가중합으로 최종 점수 계산 + 항목별 기여도 저장
    results = []
    for i, p in enumerate(products):
        breakdown = {}
        score = 0.0
        for key in nutrient_keys:
            contrib = normalized[key][i] * weights[key]
            breakdown[key] = round(contrib, 3)
            score += contrib
        results.append({"name": p.name, "score": round(score, 3), "breakdown": breakdown})

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def explain_recommendation(results: List[Dict], goal: str) -> str:
    """1위 제품에 대해 어떤 영양소 항목이 가장 크게 기여했는지 근거 문구 생성."""
    best = results[0]
    top_factor = max(best["breakdown"], key=best["breakdown"].get)
    factor_name = NUTRIENT_LABELS.get(top_factor, top_factor)
    return f"'{best['name']}'가 '{goal}' 목표에 가장 적합합니다. 특히 {factor_name} 항목에서 다른 제품 대비 유리했습니다."


# ---------------------------------------------------------------------------
# 자유 서술 건강정보 해석 (AI)
# 예: "당뇨 초기라서 혈당 관리 중이에요" -> {"sugar": 1.2, "carb": 1.0, "calorie": 0.4}
# ---------------------------------------------------------------------------

HEALTH_TEXT_PROMPT = """당신은 영양 상담 보조 AI입니다. 사용자가 자유롭게 작성한 건강 상태 설명을 읽고,
아래 영양소 항목 중 어떤 항목에 얼마나 가중치를 둬야 하는지 JSON으로만 답하세요.

사용 가능한 항목: calorie, sugar, sodium, fat, protein, cost, carb, saturated_fat, trans_fat, cholesterol

규칙:
- 가중치는 0.0 초과 1.5 이하의 숫자로, 사용자 건강 상태와 관련이 깊을수록 높게 설정하세요.
- 관련 없는 항목은 아예 포함하지 마세요 (0으로 넣지 말고 키 자체를 생략).
- 최소 1개, 최대 4개 항목만 포함하세요.
- 반드시 JSON 객체만 출력하고, 다른 설명이나 코드블록 표시(```)는 절대 포함하지 마세요.
- 출력 예시: {{"sugar": 1.2, "carb": 0.8, "calorie": 0.5}}

사용자 건강 상태 설명: "{user_text}"
"""

# API 호출 실패 시를 대비한 키워드 기반 폴백 규칙 (데모 중 네트워크 장애 대비용)
KEYWORD_FALLBACK_RULES = [
    (["당뇨", "혈당"], {"sugar": 1.2, "carb": 1.0, "calorie": 0.4}),
    (["고혈압", "혈압"], {"sodium": 1.5, "calorie": 0.3}),
    (["고지혈증", "콜레스테롤", "심혈관"], {"saturated_fat": 1.0, "trans_fat": 1.2, "cholesterol": 0.8}),
    (["다이어트", "체중", "살", "감량"], {"calorie": 1.0, "sugar": 0.5, "fat": 0.4}),
    (["근육", "벌크업", "증량"], {"protein": 1.2, "calorie": 0.4}),
]


def interpret_health_text(user_text: str, model: str = "gemini-flash-latest", api_key: str = None) -> Dict[str, float]:
    """
    자유 서술 건강정보를 AI(Gemini)가 해석해 영양소별 가중치 딕셔너리로 변환한다.
    api_key를 직접 넘기면 그 값을 우선 사용하고, 넘기지 않으면
    GEMINI_API_KEY 환경변수를 시도한다.
    실패(네트워크 오류, 인증 오류, JSON 파싱 실패, 유효하지 않은 값 등) 시
    빈 딕셔너리({})를 반환한다. 상위 함수(get_weights_from_health_text)가
    이 경우 폴백 로직으로 넘어간다.
    """
    try:
        from google import genai  # 지연 import: 라이브러리 없이도 나머지 코드는 동작하게

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError("GEMINI_API_KEY가 설정되지 않았고, api_key 인자로도 전달되지 않았습니다.")

        client = genai.Client(api_key=resolved_key)
        response = client.models.generate_content(
            model=model,
            contents=HEALTH_TEXT_PROMPT.format(user_text=user_text),
        )
        raw_text = response.text.strip()
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        weights = {k: float(v) for k, v in json.loads(raw_text).items()}
        return validate_weights(weights)
    except Exception as e:
        print(f"[경고] AI 건강정보 해석 실패, 폴백으로 전환합니다: {e}")
        return {}


def keyword_fallback(user_text: str) -> Dict[str, float]:
    """AI 해석이 실패했을 때 사용하는 단순 키워드 매칭 폴백."""
    for keywords, weights in KEYWORD_FALLBACK_RULES:
        if any(kw in user_text for kw in keywords):
            return weights
    return {}


def get_weights_from_health_text(user_text: str, api_key: str = None, model: str = "gemini-flash-latest") -> Dict[str, float]:
    """
    자유 서술 건강정보 -> 가중치 딕셔너리 변환의 진입점.
    1) AI 해석 시도 -> 2) 실패 시 키워드 폴백 -> 3) 그마저 실패 시 에러
    반환된 딕셔너리는 score_products(custom_weights=...)에 바로 넘기면 된다.
    """
    weights = interpret_health_text(user_text, model=model, api_key=api_key)
    if weights:
        return weights

    fallback = keyword_fallback(user_text)
    if fallback:
        print("[안내] 키워드 기반 규칙으로 가중치를 대체했습니다.")
        return fallback

    raise ValueError(
        "건강 상태 설명에서 관련 영양 기준을 찾지 못했습니다. "
        "프리셋 목표(goal)를 선택하시거나 직접 가중치를 입력해주세요."
    )


if __name__ == "__main__":
    # 간단한 동작 확인용 예시 (과자 3종 비교, cost는 1회 제공량당 가격으로 환산된 값)
    sample_products = [
        Product("과자 A", {
            "calorie": 250, "sugar": 12, "sodium": 450, "protein": 3, "fat": 10, "cost": 800,
            "carb": 30, "saturated_fat": 4, "trans_fat": 0.3, "cholesterol": 5,
        }),
        Product("과자 B", {
            "calorie": 180, "sugar": 5, "sodium": 300, "protein": 5, "fat": 6, "cost": 1200,
            "carb": 20, "saturated_fat": 1.5, "trans_fat": 0.0, "cholesterol": 0,
        }),
        Product("과자 C", {
            "calorie": 300, "sugar": 20, "sodium": 600, "protein": 2, "fat": 15, "cost": 600,
            "carb": 35, "saturated_fat": 6, "trans_fat": 0.5, "cholesterol": 10,
        }),
    ]

    for goal in ("weight_loss", "muscle", "blood_sugar", "anti_swelling"):
        print(f"\n=== 목표: {goal} ===")
        ranked = score_products(sample_products, goal)
        for r in ranked:
            print(r)
        print(explain_recommendation(ranked, goal))

    # 사용자가 프리셋 없이 직접 가중치를 설정한 경우 (미세조정 시나리오)
    print("\n=== 사용자 커스텀: 나트륨은 매우 중요, 가격은 조금 신경 씀 ===")
    my_weights = {"sodium": 1.0, "cost": 0.3}
    ranked = score_products(sample_products, custom_weights=my_weights)
    for r in ranked:
        print(r)
    print(explain_recommendation(ranked, "내가 설정한 기준"))

    # 자유 서술 건강정보 해석 시나리오
    print("\n=== 자유 서술 건강정보: '당뇨 초기라서 혈당 관리 중이에요' ===")
    health_text = "당뇨 초기라서 혈당 관리 중이에요"
    weights_from_text = get_weights_from_health_text(health_text)
    print(f"해석된 가중치: {weights_from_text}")
    ranked = score_products(sample_products, custom_weights=weights_from_text)
    for r in ranked:
        print(r)
    print(explain_recommendation(ranked, "혈당 관리"))