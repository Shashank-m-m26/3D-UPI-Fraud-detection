import streamlit as st
import sys
import os
import plotly.graph_objects as go
import pandas as pd
import time

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from risk.risk_engine import score_transaction


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="UPI Risk Engine", layout="wide")

# -----------------------------
# STYLING (FINTECH LOOK)
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.card {
    padding: 15px;
    border-radius: 12px;
    background: #161B22;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# GAUGE FUNCTION
# -----------------------------
def risk_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        title={'text': "Risk Level (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "white"},
            'steps': [
                {'range': [0, 30], 'color': "#00ff9f"},
                {'range': [30, 70], 'color': "#ffa500"},
                {'range': [70, 100], 'color': "#ff4b4b"},
            ],
        }
    ))

    fig.update_layout(
        paper_bgcolor="#0E1117",
        font={'color': "white"}
    )

    return fig


# -----------------------------
# HEADER
# -----------------------------
st.title("🏦 UPI Fraud Risk Monitoring System")
st.caption("Real-time Post-Authentication Fraud Detection")

st.markdown("---")


# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.title("⚙️ Transaction Simulator")

amount = st.sidebar.number_input("💰 Amount", value=5000)
hour = st.sidebar.slider("⏰ Hour", 0, 23, 12)

new_device = st.sidebar.selectbox("📱 New Device", [0, 1])
new_receiver = st.sidebar.selectbox("👤 New Receiver", [0, 1])

amount_spike = st.sidebar.slider("📈 Amount Spike", 0.5, 5.0, 1.0)
burst_flag = st.sidebar.selectbox("⚡ Burst Transaction", [0, 1])
fraud_receiver = st.sidebar.selectbox("🚨 Suspicious Receiver", [0, 1])

network_cluster_size = st.sidebar.slider("🌐 Network Risk", 0.0, 1.0, 0.2)
receiver_risk_score = st.sidebar.slider("📊 Receiver Risk Score", 0.0, 1.0, 0.2)

night_tx = 1 if hour < 6 else 0


# -----------------------------
# SESSION STATE (for timeline)
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# -----------------------------
# MAIN LAYOUT
# -----------------------------
col1, col2, col3 = st.columns([1, 1, 2])


# -----------------------------
# RUN MODEL
# -----------------------------
if st.sidebar.button("🚀 Analyze Transaction"):

    features = {
        "amount": amount,
        "hour": hour,
        "new_device": new_device,
        "night_tx": night_tx,
        "amount_spike": amount_spike,
        "new_receiver": new_receiver,
        "burst_flag": burst_flag,
        "fraud_receiver": fraud_receiver,
        "network_cluster_size": network_cluster_size,
        "receiver_risk_score": receiver_risk_score
    }

    result = score_transaction(features)

    risk = result["risk_score"]
    decision = result["decision"]
    reasons = result["reasons"]

    # Save to timeline
    st.session_state.history.append({
        "time": time.strftime("%H:%M:%S"),
        "amount": amount,
        "risk": risk,
        "decision": decision
    })

    # -----------------------------
    # COL 1: GAUGE
    # -----------------------------
    with col1:
        st.subheader("📊 Risk Meter")
        st.plotly_chart(risk_gauge(risk), use_container_width=True)

    # -----------------------------
    # COL 2: DECISION
    # -----------------------------
    with col2:
        st.subheader("🧠 Decision")

        if decision == "ALLOW":
            st.success("✅ ALLOW")
        elif decision == "REVIEW":
            st.warning("⚠️ REVIEW")
        else:
            st.error("🚫 BLOCK")

        st.markdown(f"### Score: {risk:.3f}")

    # -----------------------------
    # COL 3: EXPLANATION
    # -----------------------------
    with col3:
        st.subheader("📌 Risk Drivers")

        if reasons:
            for r in reasons:
                st.markdown(f"""
                <div class="card">
                🔸 {r}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="card">
            ✅ No major risk signals
            </div>
            """, unsafe_allow_html=True)


# -----------------------------
# FRAUD TIMELINE PANEL
# -----------------------------
st.markdown("---")
st.subheader("📈 Transaction Timeline")

if st.session_state.history:

    df = pd.DataFrame(st.session_state.history)

    st.dataframe(df, use_container_width=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["risk"],
        mode='lines+markers',
        name="Risk Score"
    ))

    fig.update_layout(
        title="Risk Trend Over Time",
        xaxis_title="Time",
        yaxis_title="Risk Score",
        paper_bgcolor="#0E1117",
        font={'color': "white"}
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No transactions yet. Run simulation to see timeline.")


# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Model: XGBoost + Hybrid Risk Engine | Fintech Fraud Detection System")