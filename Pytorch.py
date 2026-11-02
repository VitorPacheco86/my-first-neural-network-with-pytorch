import torch
from torch import nn

# Seleciona GPU (CUDA), se disponível
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Dispositivo:", device)

# Dados de entrada: horas de estudo, frequência, nota anterior e horas de sono
x = torch.tensor([
    [1, 40, 3, 5],
    [2, 50, 4, 6],
    [3, 60, 5, 7],
    [4, 70, 6, 7],
    [5, 80, 7, 8],
    [6, 90, 8, 8],
    [7, 95, 9, 9],
    [2, 30, 2, 4],
    [8, 100, 10, 9],
    [1, 20, 2, 5]
], dtype=torch.float32, device=device)

# Normalização simples dos dados
x = x / 100

# Rótulos: 0 = reprovado, 1 = aprovado
y = torch.tensor([
    [0], [0], [0], [1], [1],
    [1], [1], [0], [1], [0]
], dtype=torch.float32, device=device)

# Arquitetura da rede neural
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
).to(device)

# Função de perda e otimizador
criterion = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

# Treinamento
epochs = 1000

for i in range(epochs):
    pred = model(x)
    loss = criterion(pred, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if i % 100 == 0:
        print(f"Época: {i:4d} | Loss: {loss.item():.6f}")

# Avaliação
model.eval()

with torch.no_grad():
    logits = model(x)
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).int()
    accuracy = (predictions == y.int()).float().mean()

print("\nProbabilidades:")
print(probabilities)

print("\nClassificações:")
print(predictions)

print(f"\nAcurácia nos dados de treinamento: {accuracy.item() * 100:.2f}%")
