from dataclasses import dataclass

@dataclass
class BasicTransformerConfig():
    width: int = 468

    heads_num: int = 2
    blocks_num: int = 6

    dropout: float = 0.1


    window_size: int = 8192 // 8
    shift: int = 1 # this should be 1 when using autoregressive wrapper

    data_path: str = '/root/secret_sauce/savant-32000-compressed.pt'
    
    
    
    
    # dead param for basic transformer
    ff_dim: int = 512