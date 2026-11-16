"""
server.py
nutrition_scoring.py의 로직을 Node.js(또는 다른 클라이언트)가 호출할 수 있도록
HTTP API로 노출하는 서버.

실행:
    pip install flask
    python server.py
    (기본적으로 http://localhost:5000 에서 뜬다)
"""

from flask import Flask, request, jsonify
import os

from nutrition_scoring import (
    Product,
    score_products,
    explain_recommendation,
    get_weights_from_health_text,
    GOAL_WEIGHTS,
)

app = Flask(__name__)

INTERNAL_API_TOKEN = os.environ.get("INTERNAL_API_TOKEN")

@app.before_request
def check_internal_token():
    # /health 같은 헬스체크 엔드포인트가 있다면 예외 처리하고 싶을 때 여기서 분기
    if not INTERNAL_API_TOKEN:
        # 토큰 자체를 설정 안 했으면 개발 환경으로 보고 통과 (배포 전엔 반드시 설정할 것)
        return
    token = request.headers.get("X-Internal-Token")
    if token != INTERNAL_API_TOKEN:
        return jsonify({"error": "인증되지 않은 요청입니다."}), 401

@app.route("/recommend", methods=["POST"])
def recommend():
    """
    요청 바디 예시:
    {
      "products": [
        {"name": "과자 A", "nutrients": {"calorie": 250, "sugar": 12, "sodium": 450}},
        {"name": "과자 B", "nutrients": {"calorie": 180, "sugar": 5, "sodium": 300}}
      ],
      "goal": "weight_loss",          // 프리셋 사용 시
      "health_text": "당뇨 초기예요",   // 자유 서술 사용 시 (goal 대신 이거 하나만 보내도 됨)
      "custom_weights": {"sugar": 1.0} // 직접 가중치 지정 시 (최우선 적용)
    }

    응답 예시:
    {
      "weights_used": {...},
      "ranked": [...],
      "recommendation": "..."
    }
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "요청 바디가 비어있거나 JSON 형식이 아닙니다."}), 400

    raw_products = body.get("products")
    if not raw_products or not isinstance(raw_products, list):
        return jsonify({"error": "products는 최소 1개 이상의 배열이어야 합니다."}), 400

    try:
        products = [Product(name=p["name"], nutrients=dict(p["nutrients"])) for p in raw_products]
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"products 형식이 올바르지 않습니다: {e}"}), 400

    goal = body.get("goal")
    health_text = body.get("health_text")
    custom_weights = body.get("custom_weights")

    # 우선순위: custom_weights > health_text(AI 해석) > goal 프리셋
    try:
        if custom_weights:
            weights_used = custom_weights
        elif health_text:
            weights_used = get_weights_from_health_text(health_text)
        elif goal:
            if goal not in GOAL_WEIGHTS:
                return jsonify({"error": f"정의되지 않은 goal입니다: {goal}"}), 400
            weights_used = GOAL_WEIGHTS[goal]
        else:
            return jsonify({"error": "goal, health_text, custom_weights 중 하나는 반드시 필요합니다."}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # 값이 빠진 항목은 score_products가 비교에서 제외한다.
    # 남는 항목이 없으면 ValueError -> 422로 돌려서 프론트가 "직접 입력" 안내를 띄우게 한다.
    try:
        ranked, meta = score_products(products, custom_weights=weights_used, return_meta=True)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422

    label = health_text or goal or "custom"
    recommendation = explain_recommendation(ranked, label, excluded=meta["excluded"])

    return jsonify({
        "weights_used": meta["weights_used"],   # 실제로 점수에 쓴 가중치
        "requested_weights": weights_used,      # 요청한 가중치 (제외 전)
        "excluded": meta["excluded"],           # 값이 없어 비교에서 뺀 항목
        "ranked": ranked,
        "recommendation": recommendation,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
