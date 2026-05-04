🏦 UPI Fraud Detection System

⚡ A 3D Risk Engine (Device • Data • Behavior)

💡 Core Idea (What “3D” means)
This project uses a 3D risk approach to detect fraud in UPI transactions:
📱 Device Dimension
Checks if the transaction is coming from a trusted or new device
📊 Data Dimension
Looks at transaction details like amount, receiver, and timing
🧠 Behavior Dimension
Compares current activity with the user’s normal behavior

👉 Fraud is flagged when multiple dimensions show anomalies together, not just one.

🎯 What the system does
Evaluates transactions after login/OTP
Generates a risk score (0–1)
Outputs:
✅ ALLOW
⚠️ REVIEW
🚫 BLOCK
Explains why the transaction is risky

⚙️ How it works (simple)
Transaction → Features → ML Model → Rule Engine → Risk Score → Decision
ML detects patterns
Rules handle edge cases
Combined → more reliable system

🔍 Fraud it can detect
Social engineering scams
Account takeover (new device)
Suspicious receivers
Burst transactions

🖥️ UI (Dashboard)
Interactive transaction simulator
Risk meter (gauge)
Decision display
Explanation of risk
Transaction history

🛠️ Tech Stack
Python
XGBoost
Streamlit
Plotly

💬 One-liner
A 3D risk-based system that detects fraudulent UPI transactions in real-time using ML and behavioral intelligence.
