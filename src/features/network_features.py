import pandas as pd
import networkx as nx


def load_data():
    return pd.read_csv("data/raw/transactions.csv")


# -----------------------------
# 1. Build Graph
# -----------------------------
def build_graph(df):
    G = nx.Graph()

    for _, row in df.iterrows():
        user = f"user_{row['user_id']}"
        receiver = f"receiver_{row['receiver_id']}"

        # 🚨 KEY FIX
        if row["fraud_receiver"] == 0:
            G.add_edge(user, receiver)
        else:
            # isolate fraud receivers slightly
            G.add_node(receiver)

    return G


# -----------------------------
# 2. Cluster Size (Fraud Rings)
# -----------------------------
def compute_cluster_size(G):
    cluster_sizes = {}

    for component in nx.connected_components(G):
        size = len(component)
        for node in component:
            cluster_sizes[node] = size

    return cluster_sizes


# -----------------------------
# 3. Receiver Risk Score
# -----------------------------
def compute_receiver_risk(df):
    receiver_fraud_rate = df.groupby("receiver_id")["is_fraud"].mean()

    return receiver_fraud_rate.to_dict()


# -----------------------------
# 4. Attach Features
# -----------------------------
def add_network_features(df, cluster_sizes, receiver_risk):
    
    def get_cluster(row):
        node = f"receiver_{row['receiver_id']}"
        return cluster_sizes.get(node, 1)

    def get_receiver_risk(row):
        return receiver_risk.get(row["receiver_id"], 0)

    df["network_cluster_size"] = df.apply(get_cluster, axis=1)
    df["receiver_risk_score"] = df.apply(get_receiver_risk, axis=1)
    df["network_cluster_size"] = df["network_cluster_size"] / df["network_cluster_size"].max()
    return df


def main():
    print("Building network features...")

    df = load_data()

    G = build_graph(df)

    cluster_sizes = compute_cluster_size(G)
    receiver_risk = compute_receiver_risk(df)

    df = add_network_features(df, cluster_sizes, receiver_risk)

    df.to_csv("data/processed/network_enriched.csv", index=False)

    print("Saved to data/processed/network_enriched.csv")
    print(df[["network_cluster_size", "receiver_risk_score"]].head())


if __name__ == "__main__":
    main()