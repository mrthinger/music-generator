from dataclasses import dataclass

@dataclass
class BasicTransformerConfig():
    width: int = 2048
    ff_dim: int = 2048

    heads_num: int = 4
    blocks_num: int = 24

    dropout: float = 0.0


    window_size: int = 256
    shift: int = 1

    data_path: str = '/root/secret_sauce/savant-32000-compressed.pt'