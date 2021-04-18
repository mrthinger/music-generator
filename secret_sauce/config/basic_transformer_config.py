from dataclasses import dataclass

@dataclass
class BasicTransformerConfig():
    ff_dim: int = 2048

    heads_num: int = 4
    blocks_num: int = 4

    dropout: int = 0.0