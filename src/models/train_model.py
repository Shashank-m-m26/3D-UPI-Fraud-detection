import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib


# -----------------------------
# 1. Load Data
# -----------------------------
def load_data():
    df_feat = pd.read_csv("data/processed/transactions_features.csv")
    df_net = pd.read_csv("data/processed/network_enriched.csv")

    # merge on index (since both come from same base df)
    df = pd.concat([df_feat, df_net[[
        "network_cluster_size",
        "receiver_risk_score"
    ]]], axis=1)

    return df


# -----------------------------
# 2. Prepare Features
# -----------------------------
def prepare_data(df):
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


# -----------------------------
# 3. Train Model
# -----------------------------
def train_model(X_train, y_train):
    model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train)
    return model


# -----------------------------
# 4. Evaluate Model
# -----------------------------
def evaluate(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.2).astype(int)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))


# -----------------------------
# 5. Save Model
# -----------------------------
def save_model(model):
    joblib.dump(model, "models/xgb_model.pkl")
    print("Model saved to models/xgb_model.pkl")


def main():
    print("Training model...")

    df = load_data()
    print(df.corr()["is_fraud"].sort_values(ascending=False))
    X_train, X_test, y_train, y_test = prepare_data(df)

    print(df["network_cluster_size"].describe())

    model = train_model(X_train, y_train)

    evaluate(model, X_test, y_test)

    save_model(model)


if __name__ == "__main__":
    main()