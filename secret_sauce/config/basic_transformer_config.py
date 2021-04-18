from dataclasses import dataclass

@dataclass
class BasicTransformerConfig():
    ff_dim: int = 2048

    heads_num: int = 1
    blocks_num: int = 8

    dropout: float = 0.0