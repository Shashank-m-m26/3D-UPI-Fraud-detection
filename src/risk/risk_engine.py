import joblib
import numpy as np


# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("models/xgb_model.pkl")


# -----------------------------
# Decision Logic
# -----------------------------
def get_decision(prob):
    if prob < 0.3:
        return "ALLOW"
    elif prob < 0.7:
        return "REVIEW"
    else:
        return "BLOCK"


# -----------------------------
# Explainability
# -----------------------------
def explain(features):
    reasons = []

    if features["amount_spike"] > 2:
        reasons.append("Unusually high transaction amount")

    if features["new_device"] == 1:
        reasons.append("New device detected")

    if features["night_tx"] == 1:
        reasons.append("Transaction at unusual hour")

    if features["fraud_receiver"] == 1:
        reasons.append("Receiver linked to suspicious activity")

    if features["burst_flag"] == 1:
        reasons.append("Multiple transactions in short time")

    if features["network_cluster_size"] > 0.8:
        reasons.append("Connected to large suspicious network")

    return reasons


# -----------------------------
# Rule Engine (🔥 NEW)
# -----------------------------
def apply_rules(prob, f):

    # Rule 1: Strong Social Engineering
    if (
        f["fraud_receiver"] == 1 and
        f["new_receiver"] == 1 and
        f["night_tx"] == 1 and
        f["amount"] >= 20000
    ):
        prob = max(prob, 0.6)

    # Rule 2: Account Takeover
    if (
        f["new_device"] == 1 and
        f["amount_spike"] > 2
    ):
        prob = max(prob, 0.5)

    # Rule 3: Burst Fraud
    if f["burst_flag"] == 1:
        prob = max(prob, prob + 0.2)

    # Rule 4: High-risk Receiver Alone
    if f["receiver_risk_score"] > 0.5:
        prob = max(prob, 0.4)

    return min(prob, 0.95)  # cap probability


# -----------------------------
# Main Risk Scoring Function
# -----------------------------
def score_transaction(feature_dict):

    feature_order = [
        "amount",
        "hour",
        "new_device",
        "night_tx",
        "amount_spike",
        "new_receiver",
        "burst_flag",
        "fraud_receiver",
        "network_cluster_size",
        "receiver_risk_score"
    ]

    X = np.array([feature_dict[f] for f in feature_order]).reshape(1, -1)

    ml_prob = model.predict_proba(X)[0][1]

    rule_score = 0

    if feature_dict["fraud_receiver"] == 1:
        rule_score += 0.3

    if feature_dict["new_receiver"] == 1:
        rule_score += 0.2

    if feature_dict["night_tx"] == 1:
        rule_score += 0.15

    if feature_dict["amount_spike"] > 2:
        rule_score += 0.2

    if feature_dict["burst_flag"] == 1:
        rule_score += 0.25

    if feature_dict["receiver_risk_score"] > 0.5:
        rule_score += 0.2

    # Combine ML + rules
    prob = 0.5 * ml_prob + 0.5 * rule_score

    # Apply overrides
    prob = apply_rules(prob, feature_dict)
    prob = min(prob, 0.95)

    # Decision + explanation
    decision = get_decision(prob)
    reasons = explain(feature_dict)

    return {
        "risk_score": round(float(prob), 3),
        "decision": decision,
        "reasons": reasons
    }


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":
    sample = {
        "amount": 50000,
        "hour": 2,
        "new_device": 0,
        "night_tx": 1,
        "amount_spike": 1.3,
        "new_receiver": 1,
        "burst_flag": 0,
        "fraud_receiver": 1,
        "network_cluster_size": 0.18,
        "receiver_risk_score": 0.38
    }

    result = score_transaction(sample)
    print(result)