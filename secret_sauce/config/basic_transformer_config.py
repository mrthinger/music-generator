from dataclasses import dataclass

@dataclass
class BasicTransformerConfig():
    width: int = 800

    heads_num: int = 4
    blocks_num: int = 18

    dropout: float = 0.1


    window_size: int = 8192 // 2
    shift: int = 1 # this should be 1 when using autoregressive wrapper

    data_path: str = '/root/secret_sauce/savant-32000-compressed.pt'
    
    
    
    
    # dead param for basic transformer
    ff_dim: int = 512