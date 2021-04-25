from dataclasses import dataclass

@dataclass
class VQVAEConfig:
    embedding_dim: int = 64
    num_embeddings: int = 2048
    
    VQ_commit_loss_beta: float = 0.02
    VQ_decay: float = .75
