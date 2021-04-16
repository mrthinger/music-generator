from dataclasses import dataclass

@dataclass
class VQVAEConfig:
    embedding_dim: int = 64
    num_embeddings: int = 2048