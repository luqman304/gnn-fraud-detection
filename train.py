import numpy as np, pandas as pd, networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

np.random.seed(42)
# Build synthetic bipartite graph of users and merchants
n_users, n_merch = 300, 80
G = nx.Graph()
for u in range(n_users):
    G.add_node(f"u{u}", type="user", feat=np.random.rand(4))
for m in range(n_merch):
    G.add_node(f"m{m}", type="merch", feat=np.random.rand(4))

edges = []
for u in range(n_users):
    for _ in range(np.random.poisson(5)):
        m = np.random.randint(0, n_merch)
        amt = np.random.exponential(50)
        G.add_edge(f"u{u}", f"m{m}", amount=amt)
        edges.append((f"u{u}", f"m{m}", amt))

# Create node features: original feat + degree and avg amount
X, y = [], []
for n, data in G.nodes(data=True):
    nbrs = list(G.neighbors(n))
    deg = len(nbrs)
    if deg == 0: avg_amt = 0.0
    else:
        avg_amt = np.mean([G[n][v]["amount"] for v in nbrs if "amount" in G[n][v]] or [0.0])
    feat = np.concatenate([data["feat"], [deg, avg_amt]])
    X.append(feat)
    # fraud label: rare for users with high degree and high avg amount (synthetic rule)
    label = 1 if (data["type"]=="user" and deg>8 and avg_amt>70) else 0
    y.append(label)

X = np.asarray(X); y = np.asarray(y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
pred = clf.predict(Xte)
print(classification_report(yte, pred, digits=3))
print("Done. Next: replace classifier with GraphSAGE in torch-geometric.")
