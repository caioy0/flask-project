import os
from dotenv import load_dotenv # Carrega o env
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import pandas as pd

load_dotenv() # Carrega o env

# Obter o caminho do arquivo do ambiente
base_path = Path(__file__).parent.parent
file_path = base_path / os.getenv("FILE_PATH")

df = pd.read_excel(file_path, engine='openpyxl') # Carrega dados da planilha

# Selecionar as colunas que contêm os números das bolas
colunas_bolas = ['C', 'D', 'E', 'F', 'G', 'H']  # Ajuste os nomes se necessário
numbers = df[colunas_bolas].values  # Extrai os valores numéricos dessas colunas

# Criar uma representação numérica de cada número
le = LabelEncoder()
numbers_encoded = le.fit_transform(numbers)

# Converter para um tensor do PyTorch
numbers_tensor = torch.tensor(numbers_encoded, dtype=torch.long)

#Classes
class NumberDataset(Dataset):
    def __init__(self, numbers, seq_length=10):
        self.numbers = numbers
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.numbers) - self.seq_length
    
    def __getitem__(self, idx):
        input_seq = self.numbers[idx:idx+self.seq_length]
        target = self.numbers[idx+self.seq_length]
        return torch.tensor(input_seq, dtype=torch.long), target

# Definir os dados de entrada e saída (sequência de números)
dataset = NumberDataset(numbers_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class NumberPredictionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=10, hidden_dim=50):
        super(NumberPredictionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])  # Pegando apenas a última saída da sequência
        return output

# Inicializando o modelo
vocab_size = len(le.classes_)  # Número total de diferentes números
model = NumberPredictionModel(vocab_size)

# Definir a função de perda e o otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Treinamento
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, target in dataloader:
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")


# Fazendo previsões
model.eval()  # Coloca o modelo em modo de avaliação
with torch.no_grad():
    last_numbers = numbers_tensor[-10:]  # Últimos 10 números
    input_seq = torch.tensor(last_numbers, dtype=torch.long).unsqueeze(0)  # Batch size = 1
    predicted = model(input_seq)
    predicted_number = predicted.argmax(dim=1).item()  # Pega o número com maior probabilidade
    predicted_number_label = le.inverse_transform([predicted_number])
    print(f"Próximo número previsto: {predicted_number_label[0]}")


# Debug
print(df)
print(f"Arquivo carregado de: {file_path}")
print(f"PyTorch version: {torch.__version__}")