import streamlit as st, networkx as nx, random, matplotlib.pyplot as plt

st.set_page_config(page_title="GNN Fraud Detection", page_icon="🔗", layout="wide")
st.title("Graph Neural Network Fraud Detection — Starter")

# synthetic graph
G = nx.erdos_renyi_graph(20,0.1)
fraud_nodes = random.sample(G.nodes(), 3)
colors = ["red" if n in fraud_nodes else "lightblue" for n in G.nodes()]

fig, ax = plt.subplots()
nx.draw(G, node_color=colors, with_labels=True, ax=ax)
st.pyplot(fig)

st.write("Fraudulent nodes (red):", fraud_nodes)
st.info("Next steps: implement GCN in DGL/PyTorch Geometric; train classifier for fraud detection.")
