import torch
import torch.nn as nn

class ConceptEntropyLoss(nn.Module):
    """
    Penalidade de Gargalo Semântico (Parte 1/2): Regularização de Entropia.
    Força as ativações dos conceitos contínuos (c_hat) em direção a 0 ou 1, 
    minimizando a entropia binária. Isso reduz o espaço latente disponível 
    para o modelo "esconder" informações (Concept Leakage).
    
    Matemática: L_ent = - (1/k) * sum [ c * log(c) + (1-c) * log(1-c) ]
    """
    def __init__(self, eps: float = 1e-8):
        super(ConceptEntropyLoss, self).__init__()
        self.eps = eps # Epsilon para estabilidade numérica no logaritmo

    def forward(self, c_hat: torch.Tensor) -> torch.Tensor:
        # c_hat shape esperado: (batch_size, num_concepts)
        # Assumimos que c_hat já passou por uma ativação Sigmoid (valores entre 0 e 1)
        entropy = - (c_hat * torch.log(c_hat + self.eps) + (1 - c_hat) * torch.log(1 - c_hat + self.eps))
        
        # Retorna a média sobre o batch e sobre todos os conceitos
        return entropy.mean()


class ConceptOrthogonalityLoss(nn.Module):
    """
    Penalidade de Gargalo Semântico (Parte 2/2): Penalidade de Ortogonalidade.
    Calcula a matriz de covariância empírica das ativações dos conceitos no batch
    e penaliza os elementos fora da diagonal principal. Isso evita que o modelo 
    crie "atalhos" vazando informações através da correlação espúria entre conceitos.
    
    Matemática: L_ortho = || Sigma - diag(Sigma) ||_F^2
    """
    def __init__(self):
        super(ConceptOrthogonalityLoss, self).__init__()

    def forward(self, c_hat: torch.Tensor) -> torch.Tensor:
        # c_hat shape esperado: (batch_size, num_concepts)
        batch_size = c_hat.size(0)
        
        # A covariância exige pelo menos 2 exemplos no batch
        if batch_size < 2:
            return torch.tensor(0.0, device=c_hat.device, requires_grad=True)

        # 1. Centralizar os conceitos na média (Criar a matriz Z)
        c_mean = c_hat.mean(dim=0, keepdim=True)
        Z = c_hat - c_mean

        # 2. Computar a matriz de covariância empírica (Sigma)
        # Z tem dimensão (B, K), então Z.T @ Z tem dimensão (K, K)
        cov_matrix = (Z.t() @ Z) / (batch_size - 1)

        # 3. Extrair a diagonal principal (queremos manter a variância, mas zerar a covariância)
        diag = torch.diag(torch.diag(cov_matrix))

        # 4. Isolar os elementos fora da diagonal principal
        off_diagonal = cov_matrix - diag

        # 5. Penalizar aplicando a norma de Frobenius ao quadrado (soma dos quadrados)
        loss = torch.sum(off_diagonal ** 2)
        
        return loss


class CBMLoss(nn.Module):
    """
    Loss Total do Concept Bottleneck Model, unindo a tarefa final, a fidelidade 
    aos conceitos humanos e as penalidades de vazamento (Leakage).
    """
    def __init__(
        self, 
        lambda_concept: float = 1.0, 
        lambda_ent: float = 0.1, 
        lambda_ortho: float = 0.1
    ):
        super(CBMLoss, self).__init__()
        
        # Losses Padrão do CBM
        self.task_loss_fn = nn.CrossEntropyLoss()
        self.concept_loss_fn = nn.BCELoss() # Supõe c_hat no intervalo [0, 1] (Sigmoid)
        
        # Novas Losses de Mitigação de Vazamento (Contribuição do Artigo)
        self.entropy_loss_fn = ConceptEntropyLoss()
        self.ortho_loss_fn = ConceptOrthogonalityLoss()

        # Pesos dos hiperparâmetros (lambdas)
        self.lambda_concept = lambda_concept
        self.lambda_ent = lambda_ent
        self.lambda_ortho = lambda_ortho

    def forward(
        self, 
        y_hat: torch.Tensor, 
        y_true: torch.Tensor, 
        c_hat: torch.Tensor, 
        c_true: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Calcula a loss total e retorna um dicionário com os componentes 
        separados para logging (ex: no Weights & Biases).
        """
        # 1. Loss da Tarefa Principal (ex: Classificação da Doença)
        task_loss = self.task_loss_fn(y_hat, y_true)

        # 2. Loss de Fidelidade dos Conceitos (Ground truth do especialista vs Previsto)
        # Garante que c_true seja float para o BCELoss
        concept_loss = self.concept_loss_fn(c_hat, c_true.float())

        # 3. Penalidades de Vazamento (Leakage)
        ent_loss = self.entropy_loss_fn(c_hat)
        ortho_loss = self.ortho_loss_fn(c_hat)

        # 4. Agregação Final L_total
        total_loss = task_loss + \
                     (self.lambda_concept * concept_loss) + \
                     (self.lambda_ent * ent_loss) + \
                     (self.lambda_ortho * ortho_loss)

        # Dicionário prático para jogar direto no wandb.log()
        metrics_dict = {
            "loss/total": total_loss.item(),
            "loss/task_classification": task_loss.item(),
            "loss/concept_fidelity": concept_loss.item(),
            "loss/leakage_entropy": ent_loss.item(),
            "loss/leakage_ortho": ortho_loss.item()
        }

        return total_loss, metrics_dict
