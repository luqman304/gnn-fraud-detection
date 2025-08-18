# GNN Fraud Detection 🔗💳  

An AI project that models transactions as a **graph** and uses Graph Neural Networks (GNNs) to detect fraudulent behavior.  
Fraud is often a **relational problem** (linked accounts, merchants, or devices), making GNNs an ideal approach.  

---

## 🚀 Features  
- 🔗 **Graph-based modeling** of user–merchant transactions  
- 🧮 **Synthetic transaction graph** for demo purposes  
- 📊 **Baseline Logistic Regression classifier** on engineered features  
- 🤖 **Upgrade path to GNNs** (GraphSAGE, GAT, GCN with PyTorch Geometric or DGL)  
- 📈 **Evaluation metrics** (precision, recall, F1, ROC AUC)  

---

## 🛠️ Tech Stack  
- **Python 3.9+**  
- **NetworkX** — synthetic graph generation & feature engineering  
- **Scikit-learn** — baseline classification  
- **Numpy/Pandas** — data handling  
- **PyTorch Geometric / DGL** (optional, for GNN upgrade)  

---

## 📦 Installation  
Clone the repo and install dependencies:  
```bash
git clone https://github.com/luqman304/gnn-fraud-detection.git
cd gnn-fraud-detection
pip install -r requirements.txt
```  

---

## ▶️ Usage  
1. Run the baseline model on a synthetic graph:  
   ```bash
   python train.py
   ```  
2. View classification metrics in the terminal.  
3. Extend to a GNN by integrating PyTorch Geometric and replacing the baseline model.  

---

## 📁 Project Structure  
```
gnn-fraud-detection/
│── train.py             # Baseline fraud detection model  
│── data/                # Placeholder for real transaction data  
│── requirements.txt     # Dependencies  
│── README.md            # Project description  
```  

---

## 🔮 Upgrade Ideas  
- Replace synthetic graph with **real transaction data**  
- Train a **Graph Neural Network** (GraphSAGE, GCN, GAT) with PyTorch Geometric  
- Add **node embeddings** for downstream fraud prediction  
- Integrate with a **real-time fraud detection pipeline** (Kafka + API)  

---

## 📌 Example Use Cases  
- 🏦 **Banking**: detect fraudulent accounts or cards  
- 🛒 **E-commerce**: identify suspicious buyers/sellers  
- 🌐 **Social platforms**: flag fake accounts and spam rings  

---

## 📜 License  
© 2025 Luqman Abdulwahab.  
This project is proprietary. No part may be used, modified, or distributed without explicit permission.  
