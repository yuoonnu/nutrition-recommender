"""
eval_recommend.py — 추천(점수 계산) 로직 평가

OCR 정답지의 실제 제품 값을 100g(ml)당으로 환산해, 같은 카테고리끼리
2~4개짜리 비교 묶음을 무작위로 만들고 nutrition_scoring.py를 직접 호출해 평가한다.
Flask 서버도, Gemini도 부르지 않는다 (비용 0원).
  ※ --with-ai 를 붙이면 자유 서술 해석(Gemini)도 평가한다 (문장 20개, 소량 비용).

실행 (nutrition_scoring.py와 같은 폴더에서):
  pip install openpyxl
  python eval_recommend.py --labels ../nutrition-vision-main/test/dataset/정답엑셀.xlsx
  python eval_recommend.py --labels ... --split test --sets 300 --with-ai

측정 지표
  1) 개선율        : Best Pick이 묶음의 나머지 제품 평균보다 목표 영양소를 얼마나 개선했는가
  2) 지배 위반     : A가 목표 항목 전부에서 B보다 좋거나 같은데 B가 더 높은 순위 → 0건이어야 정상
  3) 빈 값 안전성  : 못 읽은 값(None)이 있는 항목은 비교에서 빠지는가 (빈 값 덕분에 이득 0건)
  4) OCR 오차 민감도: 값 하나가 ±5% 틀렸을 때 1위가 바뀌는 비율
  5) 자유 서술 해석 : 키워드 폴백(항상) / Gemini 해석(--with-ai) 이 기대한 영양소를 가중치에 넣는가
  6) 제약 준수율    : 식약처 영양성분 강조표시 기준(저당·저나트륨 등)을 만족하는 제품이 묶음에 있을 때,
                     1위가 그 기준을 만족하는 비율
  7) 다양성/커버리지 : 1위로 뽑힌 서로 다른 제품 수, 한 번이라도 1위가 된 제품 비율, 카테고리별 추천 가능 비율
"""

import argparse
import json
import random
import statistics
from collections import defaultdict

from openpyxl import load_workbook

from nutrition_scoring import (
    GOAL_WEIGHTS,
    LOWER_IS_BETTER,
    NUTRIENT_LABELS,
    Product,
    interpret_health_text,
    keyword_fallback,
    score_products,
)

NUTRIENTS = ["calorie", "carb", "sugar", "protein", "fat", "saturated_fat", "trans_fat", "cholesterol", "sodium"]
GOAL_KO = {"weight_loss": "체중감량", "muscle": "가성비 근성장", "blood_sugar": "혈당 케어", "anti_swelling": "붓기 방지"}

# 자유 서술 해석 평가용 문장과, 가중치에 반드시 들어가야 하는 항목(하나 이상 포함되면 정답)
HEALTH_TEXT_CASES = [
    ("당뇨 초기라서 혈당 관리 중이에요", {"sugar", "carb"}),
    ("혈당이 높다고 해서 조심하고 있어요", {"sugar", "carb"}),
    ("단 거 줄이라고 하더라고요", {"sugar"}),
    ("고혈압 약 먹고 있어요", {"sodium"}),
    ("짠 음식 피하라고 했어요", {"sodium"}),
    ("아침마다 얼굴이 많이 부어요", {"sodium"}),
    ("콜레스테롤 수치가 높게 나왔어요", {"cholesterol", "saturated_fat", "trans_fat"}),
    ("고지혈증 진단 받았어요", {"cholesterol", "saturated_fat", "trans_fat"}),
    ("심혈관 건강이 걱정돼요", {"cholesterol", "saturated_fat", "trans_fat", "sodium"}),
    ("다이어트 중이에요", {"calorie"}),
    ("살 좀 빼고 싶어요", {"calorie"}),
    ("체중 감량이 목표예요", {"calorie"}),
    ("근육 키우는 중이라 단백질이 중요해요", {"protein"}),
    ("헬스 시작해서 벌크업 하려고요", {"protein"}),
    ("운동 후에 먹을 간식 찾아요", {"protein"}),
    ("기름진 음식은 소화가 잘 안 돼요", {"fat", "saturated_fat"}),
    ("포화지방 적은 걸로 골라주세요", {"saturated_fat"}),
    ("탄수화물을 줄이는 식단을 하고 있어요", {"carb"}),
    ("저탄고지 식단 중이에요", {"carb"}),
    ("싸고 단백질 많은 거요", {"protein", "cost"}),
]


# 키워드 규칙을 만든 뒤에 따로 작성한 검증용 문장 (규칙 튜닝에 쓰지 않음).
# 위 문장들은 규칙을 만들 때 참고했으므로, 공정한 수치는 이쪽으로 보고한다.
# 가능하면 팀원이 직접 쓴 문장으로 바꿔 넣을 것.
HEALTH_TEXT_HELDOUT = [
    ("요즘 혈당 스파이크가 걱정돼요", {"sugar", "carb"}),
    ("디저트를 끊어야 한다고 들었어요", {"sugar"}),
    ("병원에서 싱겁게 먹으라고 했어요", {"sodium"}),
    ("라면 국물 때문에 다음 날 퉁퉁 부어요", {"sodium"}),
    ("건강검진에서 LDL이 높게 나왔어요", {"cholesterol", "saturated_fat", "trans_fat"}),
    ("뱃살 빼는 중이에요", {"calorie"}),
    ("바디프로필 준비 중이에요", {"calorie", "protein"}),
    ("PT 받으면서 근력 운동해요", {"protein"}),
    ("튀긴 음식은 피하고 싶어요", {"fat", "saturated_fat"}),
    ("밥이랑 빵을 줄이고 있어요", {"carb"}),
    ("학생이라 돈이 별로 없어요", {"cost"}),
    ("임신성 당뇨 판정 받았어요", {"sugar", "carb"}),
    ("신장이 안 좋아서 염분 조절 중이에요", {"sodium"}),
    ("체지방률을 낮추고 싶어요", {"calorie", "fat"}),
    ("단백질 보충이 필요해요", {"protein"}),
]


# 식약처 「식품등의 표시기준」 영양성분 강조표시 '저' 기준 (100g당 / 100ml당, 미만)
# 목표별로 관련 있는 기준을 제약 조건으로 사용한다. 가성비 근성장(고단백)은 1일 기준치 비율 기준이라 제외.
MFDS_LOW = {
    "저당류":   ("sugar",   5.0,   2.5),
    "저나트륨": ("sodium",  120.0, 120.0),
    "저지방":   ("fat",     3.0,   1.5),
    "저열량":   ("calorie", 40.0,  20.0),
}
GOAL_CONSTRAINTS = {
    "weight_loss": ["저열량", "저당류"],
    "blood_sugar": ["저당류"],
    "anti_swelling": ["저나트륨", "저당류", "저지방"],
}


def meets(p, rule):
    key, solid, liquid = MFDS_LOW[rule]
    v = p["per100"].get(key)
    if v is None:
        return None
    return v < (liquid if p["liquid"] else solid)


# ── 정답지 읽기 ───────────────────────────────────────────

def cell(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s or s.upper() in ("N/A", "NA", "-", "없음"):
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def basis_grams(basis_text, serving, package):
    """표시 기준 문구로 100g당 환산의 기준 중량을 정한다 (normalize.js의 portionGrams와 같은 규칙)."""
    t = str(basis_text or "").replace(" ", "")
    if not t or t.upper() in ("N/A", "NA"):
        return None
    if "100g" in t or "100ml" in t.lower():
        return 100.0
    if "총내용량" in t or "totalcontent" in t.lower():
        return package
    return serving  # 1회 제공량 / OO(g)당 / 표시기준분량


def load_products(path, split=None):
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb["제품정답"]
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h).strip() if h else "" for h in rows[0]]
    out = []
    for r in rows[2:]:  # 1행 영문 키, 2행 한글 설명
        row = dict(zip(header, r))
        pid = str(row.get("product_id") or "").strip()
        if not pid or pid.upper().startswith("EX"):
            continue
        if split and str(row.get("split") or "").strip() != split:
            continue
        base = basis_grams(row.get("basis_text"), cell(row.get("serving_amount")), cell(row.get("package_amount")))
        if not base:
            continue  # 100g당 환산이 안 되는 제품은 비교 대상에서 제외
        per100 = {}
        for k in NUTRIENTS:
            v = cell(row.get(k))
            per100[k] = None if v is None else round(v * 100 / base, 3)
        unit = str(row.get("package_unit") or row.get("serving_unit") or "").lower()
        liquid = unit == "ml" or "ml" in str(row.get("basis_text") or "").lower()
        out.append({"id": pid, "category": str(row.get("category") or "기타").strip(), "per100": per100, "liquid": liquid})
    return out


# ── 묶음 만들기 ───────────────────────────────────────────

def make_sets(products, n_sets, rng):
    by_cat = defaultdict(list)
    for p in products:
        by_cat[p["category"]].append(p)
    cats = [c for c, ps in by_cat.items() if len(ps) >= 2]
    sets = []
    for _ in range(n_sets):
        cat = rng.choice(cats)
        size = rng.randint(2, min(4, len(by_cat[cat])))
        sets.append(rng.sample(by_cat[cat], size))
    return sets


def to_products(group, override=None):
    return [Product(name=p["id"], nutrients=dict(override.get(p["id"], p["per100"]) if override else p["per100"])) for p in group]


# ── 지표 계산 ─────────────────────────────────────────────

def improvement(group, ranked, weights_used):
    """Best Pick이 나머지 평균보다 목표 영양소를 몇 % 개선했는지 (+면 개선)."""
    best = ranked[0]["name"]
    vals = {p["id"]: p["per100"] for p in group}
    out = {}
    for k in weights_used:
        if k == "cost":
            continue
        others = [vals[i][k] for i in vals if i != best]
        mean = statistics.mean(others)
        if mean == 0:
            continue
        v = vals[best][k]
        out[k] = (mean - v) / mean * 100 if LOWER_IS_BETTER[k] else (v - mean) / mean * 100
    return out


def dominance_violations(group, ranked, weights_used):
    vals = {p["id"]: p["per100"] for p in group}
    rank = {r["name"]: i for i, r in enumerate(ranked)}
    keys = [k for k, w in weights_used.items() if w > 0 and k != "cost"]
    if not keys:
        return 0
    bad = 0
    for a in vals:
        for b in vals:
            if a == b:
                continue
            better_or_eq = all((vals[a][k] <= vals[b][k]) if LOWER_IS_BETTER[k] else (vals[a][k] >= vals[b][k]) for k in keys)
            strictly = any(vals[a][k] != vals[b][k] for k in keys)
            if better_or_eq and strictly and rank[a] > rank[b]:
                bad += 1
    return bad


def run_goal(group, goal):
    try:
        ranked, meta = score_products(to_products(group), goal, return_meta=True)
        return ranked, meta
    except ValueError:
        return None, None  # 목표 항목이 전부 비어 있어 비교 불가


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="정답 엑셀(.xlsx) 경로")
    ap.add_argument("--split", default=None, help="dev / test (생략하면 전체)")
    ap.add_argument("--sets", type=int, default=300, help="만들 비교 묶음 수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with-ai", action="store_true", help="자유 서술 해석(Gemini)도 평가")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    products = load_products(args.labels, args.split)
    sets = make_sets(products, args.sets, rng)
    print(f"\n제품 {len(products)}개로 비교 묶음 {len(sets)}개 생성 (split={args.split or '전체'}, seed={args.seed})")
    print("※ 정답지에는 가격이 없어 '가성비 근성장'은 가격 없이 단백질만으로 계산됩니다.\n")

    report = {"products": len(products), "sets": len(sets), "goals": {}}

    # 1) 개선율 + 2) 지배 위반 + 4) OCR 오차 민감도
    winners = defaultdict(set)          # goal -> 1위가 된 제품 id
    cat_total = defaultdict(int)         # 카테고리별 (묶음×목표) 수
    cat_ok = defaultdict(int)            # 그중 점수 계산이 된 수
    for goal in GOAL_WEIGHTS:
        imps = defaultdict(list)
        cons = {r: [0, 0] for r in GOAL_CONSTRAINTS.get(goal, [])}  # [준수, 대상 묶음]
        violations = 0
        skipped = 0
        flips = 0
        flip_trials = 0
        for group in sets:
            ranked, meta = run_goal(group, goal)
            cat_total[group[0]["category"]] += 1
            if ranked is None:
                skipped += 1
                continue
            cat_ok[group[0]["category"]] += 1
            best_id = ranked[0]["name"]
            winners[goal].add(best_id)
            best = next(p for p in group if p["id"] == best_id)
            for rule in cons:
                if any(meets(p, rule) for p in group):  # 기준을 만족하는 제품이 있는 묶음만 대상
                    cons[rule][1] += 1
                    cons[rule][0] += bool(meets(best, rule))
            for k, v in improvement(group, ranked, meta["weights_used"]).items():
                imps[k].append(v)
            violations += dominance_violations(group, ranked, meta["weights_used"])

            # 1위 제품이 아닌 제품의 목표 항목 하나를 ±5% 흔들어서 1위가 바뀌는지
            target = rng.choice(group)
            keys = [k for k in meta["weights_used"] if k != "cost" and target["per100"].get(k)]
            if keys:
                k = rng.choice(keys)
                noisy = dict(target["per100"])
                noisy[k] = noisy[k] * rng.choice([0.95, 1.05])
                r2, _ = score_products(to_products(group, {target["id"]: noisy}), goal, return_meta=True)
                flip_trials += 1
                flips += r2[0]["name"] != ranked[0]["name"]

        row = {
            "평가묶음": len(sets) - skipped,
            "지배위반": violations,
            "OCR오차_1위변경률": round(flips / flip_trials * 100, 1) if flip_trials else None,
            "개선율_중앙값(%)": {NUTRIENT_LABELS[k]: round(statistics.median(v), 1) for k, v in imps.items()},
            "제약준수율(%)": {r: (round(a / b * 100, 1) if b else None, b) for r, (a, b) in cons.items()},
            "1위_서로다른제품수": len(winners[goal]),
        }
        report["goals"][goal] = row
        print(f"[{GOAL_KO[goal]}] 묶음 {row['평가묶음']}개 | 지배 위반 {violations}건 | "
              f"±5% 오차 시 1위 변경 {row['OCR오차_1위변경률']}%")
        for k, v in row["개선율_중앙값(%)"].items():
            print(f"    {k}: 나머지 평균 대비 {v:+.1f}% 개선")
        for r, (rate, n) in row["제약준수율(%)"].items():
            print(f"    식약처 '{r}' 기준 준수율: {rate}% (기준 충족 제품이 있던 묶음 {n}개)")
        print(f"    1위로 뽑힌 서로 다른 제품: {len(winners[goal])}개")

    # 7) 다양성/커버리지
    all_winners = set().union(*winners.values()) if winners else set()
    report["다양성"] = {
        "전체제품": len(products),
        "한번이라도_1위": len(all_winners),
        "커버리지(%)": round(len(all_winners) / len(products) * 100, 1),
        "카테고리별_추천가능(%)": {c: round(cat_ok[c] / cat_total[c] * 100, 1) for c in sorted(cat_total)},
    }
    print(f"\n[다양성/커버리지] 제품 {len(products)}개 중 한 번이라도 1위가 된 제품 {len(all_winners)}개 "
          f"({report['다양성']['커버리지(%)']}%)")
    for c, v in report["다양성"]["카테고리별_추천가능(%)"].items():
        print(f"    {c}: 추천 계산 가능 {v}%")

    # 3) 빈 값 안전성: 무작위 제품의 목표 항목 하나를 지우고, 그 항목이 점수에서 빠지는지
    leaks = 0
    trials = 0
    for group in sets:
        goal = rng.choice(list(GOAL_WEIGHTS))
        target = rng.choice(group)
        keys = [k for k in GOAL_WEIGHTS[goal] if k != "cost"]
        k = rng.choice(keys)
        holed = dict(target["per100"])
        holed[k] = None
        try:
            ranked, meta = score_products(to_products(group, {target["id"]: holed}), goal, return_meta=True)
        except ValueError:
            continue  # 남는 항목이 없으면 계산 거부 → 안전
        trials += 1
        if k in meta["weights_used"] or any(k in r["breakdown"] for r in ranked):
            leaks += 1
    report["빈값안전성"] = {"시험": trials, "빈 값이 점수에 반영된 경우": leaks}
    print(f"\n[빈 값 안전성] {trials}회 중 빈 값이 점수에 반영된 경우 {leaks}건 (0건이어야 정상)")

    # 5) 자유 서술 해석
    def hit(weights, expected):
        return bool(set(weights) & expected)

    fb = [hit(keyword_fallback(t), e) for t, e in HEALTH_TEXT_CASES]
    report["키워드폴백"] = {"문장": len(fb), "정답": sum(fb)}
    print(f"\n[자유 서술 - 키워드 폴백] {len(fb)}문장 중 {sum(fb)}개 정답 ({sum(fb) / len(fb) * 100:.0f}%)")
    for (t, e), ok in zip(HEALTH_TEXT_CASES, fb):
        if not ok:
            print(f"    ✗ \"{t}\" → {keyword_fallback(t) or '(해석 실패)'}")

    ho = [hit(keyword_fallback(t), e) for t, e in HEALTH_TEXT_HELDOUT]
    report["키워드폴백_검증용"] = {"문장": len(ho), "정답": sum(ho)}
    print(f"[자유 서술 - 키워드 폴백, 검증용 문장] {len(ho)}문장 중 {sum(ho)}개 정답 ({sum(ho) / len(ho) * 100:.0f}%)")
    for (t, e), ok in zip(HEALTH_TEXT_HELDOUT, ho):
        if not ok:
            print(f"    ✗ \"{t}\" → {keyword_fallback(t) or '(해석 실패)'}")

    if args.with_ai:
        # Gemini만 평가한다. 실패해도 키워드 폴백으로 넘어가지 않게 interpret_health_text를 직접 부른다.
        print()
        ai_hits, ai_fail = [], 0
        for t, e in HEALTH_TEXT_CASES + HEALTH_TEXT_HELDOUT:
            w = interpret_health_text(t)  # 실패하면 {} 반환
            if not w:
                ai_fail += 1
            ok = hit(w, e)
            ai_hits.append(ok)
            print(f"    {'✓' if ok else '✗'} \"{t}\" → {w or '(Gemini 호출 실패)'}")
        report["AI해석"] = {"문장": len(ai_hits), "정답": sum(ai_hits), "호출실패": ai_fail}
        print(f"[자유 서술 - Gemini 해석] {len(ai_hits)}문장 중 {sum(ai_hits)}개 정답 ({sum(ai_hits) / len(ai_hits) * 100:.0f}%), 호출 실패 {ai_fail}건")
        if ai_fail:
            print("    ※ 호출 실패가 있으면 Gemini 성능이 아닙니다. 위 [경고] 메시지로 원인을 확인하세요.")

    with open("eval_recommend_result.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n결과 저장 → eval_recommend_result.json")


if __name__ == "__main__":
    main()
