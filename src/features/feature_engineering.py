import pandas as pd


def load_data():
    return pd.read_csv("data/raw/transactions.csv")


def sort_data(df):
    return df.sort_values(by=["user_id", "hour"]).reset_index(drop=True)


# -----------------------------
# 1. Amount Spike (Behavioral)
# -----------------------------
def add_amount_spike(df):
    user_avg = df.groupby("user_id")["amount"].transform("mean")
    df["amount_spike"] = df["amount"] / (user_avg + 1)
    return df


# -----------------------------
# 2. New Receiver
# -----------------------------
def add_new_receiver_flag(df):
    df["new_receiver"] = (
        ~df.groupby("user_id")["receiver_id"]
        .transform(lambda x: x.duplicated())
    ).astype(int)
    return df


# -----------------------------
# 3. Burst Transactions
# -----------------------------
def add_burst_flag(df):
    df["tx_count"] = (
        df.groupby("user_id")["hour"]
        .transform(lambda x: x.rolling(window=3, min_periods=1).count())
    )

    df["burst_flag"] = (df["tx_count"] >= 3).astype(int)
    return df


# -----------------------------
# 4. Final Feature Selection
# -----------------------------
def select_features(df):
    features = [
        "amount",
        "hour",
        "new_device",
        "night_tx",
        "amount_spike",
        "new_receiver",
        "burst_flag",
        "fraud_receiver"
    ]

    return df[features + ["is_fraud"]]


def main():
    print("Running feature engineering...")

    df = load_data()
    df = sort_data(df)

    df = add_amount_spike(df)
    df = add_new_receiver_flag(df)
    df = add_burst_flag(df)

    df_final = select_features(df)

    df_final.to_csv("data/processed/transactions_features.csv", index=False)

    print("Saved to data/processed/transactions_features.csv")
    print(df_final.head())


if __name__ == "__main__":
    main()