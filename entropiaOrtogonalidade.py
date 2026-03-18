import torch

# Fixando a semente para os resultados serem os mesmos na sua tela
torch.manual_seed(42)

# Simulação: Um batch de 4 imagens, prevendo 3 conceitos (ex: Halo, Lesão, Esporos)
# Perceba que propositalmente criei um "atalho": o Conceito 1 e 2 andam muito juntos.
# Quando um é alto, o outro também é. Isso é o vazamento (correlação espúria)!
c_hat = torch.tensor([
    [0.85, 0.90, 0.10],  # Imagem 1
    [0.80, 0.88, 0.15],  # Imagem 2
    [0.10, 0.15, 0.95],  # Imagem 3
    [0.15, 0.10, 0.85]   # Imagem 4
])

print("=== 1. PREVISÕES ORIGINAIS (c_hat) ===")
print("Valores intermediários (ex: 0.85) escondem dados. Queremos 1.0 ou 0.0.")
print(c_hat)


# ---------------------------------------------------------
# INVESTIGANDO A ENTROPIA
# ---------------------------------------------------------
eps = 1e-8
# Aplicando a fórmula exata da entropia binária
entropia_tensor = - (c_hat * torch.log(c_hat + eps) + (1 - c_hat) * torch.log(1 - c_hat + eps))

print("\n=== 2. RAIO-X DA ENTROPIA ===")
print("Veja como valores próximos de 0.8/0.1 geram uma penalidade de ~0.3 a ~0.4.")
print("Se c_hat fosse exatamente 0.99, a penalidade cairia para ~0.05 (ideal).")
print("O PyTorch vai tentar esmagar esses números para zero no backpropagation.")
print(entropia_tensor)


# ---------------------------------------------------------
# INVESTIGANDO A ORTOGONALIDADE E COVARIÂNCIA
# ---------------------------------------------------------
batch_size = c_hat.size(0)

# Passo A: Centralizar na média
c_mean = c_hat.mean(dim=0, keepdim=True)
print("\n=== 2.1. MÉDIA DOS CONCEITOS ===")
print(c_mean)
    
Z = c_hat - c_mean
print("\n=== 2.2. CONCEITOS CENTRALIZADOS (Z) ===")
print(Z)

# Passo B: Matriz de Covariância
cov_matrix = (Z.t() @ Z) / (batch_size - 1)

print("\n=== 3. MATRIZ DE COVARIÂNCIA (O 'Ninho' do Vazamento) ===")
print("O eixo X e Y representam os 3 conceitos (Matriz 3x3).")
print(cov_matrix)

# Passo C: Isolando o problema
diag = torch.diag(torch.diag(cov_matrix))
off_diagonal = cov_matrix - diag

print("\n=== 4. O ALVO DA NOSSA LOSS ===")
print("Nós apagamos a diagonal principal (ela ficou com zeros).")
print("Tudo o que sobrou fora da diagonal é a 'fofoca' entre os conceitos.")
print("Veja a alta correlação (0.16) entre o conceito 0 e 1! Nossa Loss vai destruir isso.")
print(off_diagonal)

# Calculando o valor final escalar da Loss
loss_ortho = torch.sum(off_diagonal ** 2)
print(f"\nValor final da Loss de Ortogonalidade que vai para o otimizador: {loss_ortho.item():.4f}")