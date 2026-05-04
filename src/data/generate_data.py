import numpy as np
import pandas as pd

np.random.seed(42)

NUM_USERS = 1000
NUM_RECEIVERS = 500
NUM_TX = 50000


def create_users():
    return pd.DataFrame({
        "user_id": range(NUM_USERS),
        "avg_amount": np.random.uniform(1000, 5000, NUM_USERS),
        "preferred_hour": np.random.randint(8, 22, NUM_USERS),
    })


def create_receivers():
    receivers = pd.DataFrame({
        "receiver_id": range(NUM_RECEIVERS)
    })

    fraud_receivers = np.random.choice(
        receivers["receiver_id"], size=50, replace=False
    )

    return receivers, fraud_receivers


def create_device_map(users):
    device_map = {}
    for user in users["user_id"]:
        device_map[user] = [
            f"{user}_dev_{i}" for i in range(np.random.randint(1, 4))
        ]
    return device_map


def simulate_transactions(users, receivers, fraud_receivers, device_map):
    transactions = []

    user_receivers = {
    user: np.random.choice(receivers["receiver_id"], size=20, replace=False)
    for user in users["user_id"]}
    for _ in range(NUM_TX):
        user = users.sample(1).iloc[0]

        user_id = user["user_id"]

        # amount
        amount = np.random.normal(user["avg_amount"], 0.5 * user["avg_amount"])
        amount = max(100, amount)

        # hour
        hour = int(np.random.normal(user["preferred_hour"], 3)) % 24

        # receiver
        if np.random.rand() < 0.3:
            receiver_id = np.random.choice(fraud_receivers)
        else:
            receiver_id = np.random.choice(user_receivers[user_id])

        # device
        if np.random.rand() < 0.85:
            device_id = np.random.choice(device_map[user_id])
            new_device = 0
        else:
            device_id = f"{user_id}_new_dev"
            new_device = 1

        transactions.append([
            user_id, receiver_id, amount, hour, device_id, new_device
        ])

    df = pd.DataFrame(transactions, columns=[
        "user_id", "receiver_id", "amount", "hour", "device_id", "new_device"
    ])

    return df


def assign_fraud_labels(df, fraud_receivers):
    df["night_tx"] = (df["hour"] < 6).astype(int)

    user_mean = df.groupby("user_id")["amount"].transform("mean")
    df["high_amount"] = (df["amount"] > 2 * user_mean).astype(int)

    df["fraud_receiver"] = df["receiver_id"].isin(fraud_receivers).astype(int)

    def compute_prob(row):
        score = 0

        if row["high_amount"] and row["fraud_receiver"]:
            score += 0.4

        if row["new_device"] and row["night_tx"]:
            score += 0.3

        if row["fraud_receiver"]:
            score += 0.2

        if row["high_amount"]:
            score += 0.1

        return min(score, 0.95)

    df["fraud_prob"] = df.apply(compute_prob, axis=1)
    df["is_fraud"] = (np.random.rand(len(df)) < df["fraud_prob"]).astype(int)

    return df


def main():
    print("Generating data...")

    users = create_users()
    receivers, fraud_receivers = create_receivers()
    device_map = create_device_map(users)

    df = simulate_transactions(users, receivers, fraud_receivers, device_map)
    df = assign_fraud_labels(df, fraud_receivers)

    df.to_csv("data/raw/transactions.csv", index=False)

    print("Done. Saved to data/raw/transactions.csv")
    print("Fraud rate:", df["is_fraud"].mean())


if __name__ == "__main__":
    main()