# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 11:11:23 2025

@author: darek
"""
# ============================================
# Proste DEMO GNN na mini-grafie "jezuickim"
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ustawienie ziarna losowego dla powtarzalności
torch.manual_seed(42)

# -------------------------------------------------
# 1. Definiujemy prosty graf
# -------------------------------------------------
# Mamy kilka węzłów:
# 0: Ignacy Loyola
# 1: Teolog
# 2: Nauczyciel
# 3: Matematyk
# 4: Misjonarz

node_names = [
    "Ignacy",
    "Teolog",
    "Nauczyciel",
    "Matematyk",
    "Misjonarz"
]

num_nodes = len(node_names)

# Krawędzie (graf nieskierowany – więc dodamy w dwie strony)
edges = [
    (0, 1),  # Ignacy – Teolog
    (0, 4),  # Ignacy – Misjonarz
    (1, 2),  # Teolog – Nauczyciel
    (2, 3),  # Nauczyciel – Matematyk
    (4, 2),  # Misjonarz – Nauczyciel
]

# Tworzymy macierz sąsiedztwa A (N x N)
A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

for i, j in edges:
    A[i, j] = 1.0
    A[j, i] = 1.0  # bo nieskierowany

# Dodajemy self-loop (węzeł jest też swoim sąsiadem)
A = A + torch.eye(num_nodes)

# -------------------------------------------------
# 2. Cechy węzłów (node features)
# -------------------------------------------------
# Załóżmy 2 cechy:
# [duchowość, naukowość]
#
# To oczywiście sfabrykowane wartości tylko do demo.

X = torch.tensor([
    [1.0, 0.2],  # Ignacy: bardziej duchowy, trochę "praktyczny"
    [0.9, 0.1],  # Teolog: najbardziej duchowy
    [0.5, 0.5],  # Nauczyciel: miks
    [0.1, 1.0],  # Matematyk: naukowość
    [0.8, 0.3],  # Misjonarz: duchowy, trochę praktyczny
], dtype=torch.float32)

in_dim = X.shape[1]

# -------------------------------------------------
# 3. Etykiety (co ma się nauczyć GNN)
# -------------------------------------------------
# Zróbmy prostą klasyfikację:
# 1 = bardziej DUCHOWY
# 0 = bardziej NAUKOWY

y = torch.tensor([
    1,  # Ignacy
    1,  # Teolog
    1,  # Nauczyciel (załóżmy że duchowy)
    0,  # Matematyk
    1,  # Misjonarz
], dtype=torch.long)

num_classes = 2

# -------------------------------------------------
# 4. Definicja bardzo prostej GNN
# -------------------------------------------------
# Idea:
#   h_agg = A @ X         # zbieramy cechy od sąsiadów
#   h1   = ReLU( W_self * X + W_neigh * h_agg )
#   out  = W_out * h1     # logity do klasyfikacji

class SimpleGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, hidden_dim)
        self.lin_neigh = nn.Linear(in_dim, hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, adj):
        # x: (N, F)
        # adj: (N, N)
        # message passing: agregacja cech sąsiadów
        neigh_agg = adj @ x          # (N, F)
        h = self.lin_self(x) + self.lin_neigh(neigh_agg)
        h = F.relu(h)
        out = self.lin_out(h)        # (N, num_classes)
        return out, h                # zwracamy też ukryte reprezentacje

hidden_dim = 8
model = SimpleGNN(in_dim, hidden_dim, num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
criterion = nn.CrossEntropyLoss()

# -------------------------------------------------
# 5. Trening GNN
# -------------------------------------------------
print("Zaczynam trening...")

for epoch in range(201):
    model.train()
    optimizer.zero_grad()

    logits, hidden = model(X, A)  # logits: (N, 2)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()
        print(f"Epoka {epoch:3d} | Loss = {loss.item():.4f} | Accuracy = {acc*100:.1f}%")

print("\nTrening zakończony.\n")

# -------------------------------------------------
# 6. Wyniki: predykcje dla każdego węzła
# -------------------------------------------------
model.eval()
with torch.no_grad():
    logits, hidden = model(X, A)
    preds = logits.argmax(dim=1)

print("Węzeł | nazwa       | etykieta_realna | predykcja | prawdopodobieństwa [naukowy, duchowy]")
print("-------------------------------------------------------------------------------")
probs = F.softmax(logits, dim=1)

for i in range(num_nodes):
    real = y[i].item()
    pred = preds[i].item()
    p = probs[i].tolist()
    print(f"{i:3d}   | {node_names[i]:10s} | {real:14d} | {pred:9d} | {p}")

# -------------------------------------------------
# 7. Ukryte reprezentacje (embeddingi) węzłów
# -------------------------------------------------
print("\nUkryte wektory (embeddingi) węzłów po jednej warstwie GNN:")
for i in range(num_nodes):
    vec = hidden[i].detach().numpy()
    print(f"{node_names[i]:10s}: {vec}")
