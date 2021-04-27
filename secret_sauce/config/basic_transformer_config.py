from dataclasses import dataclass

@dataclass
class BasicTransformerConfig():
    width: int = 1576

    heads_num: int = 4
    blocks_num: int = 36

    dropout: float = 0.0


    window_size: int = 8192 // 16
    shift: int = 1 # this should be 1 when using autoregressive wrapper

    data_path: str = '/root/secret_sauce/savant-32000-compressed.pt'
    
    
    
    
    # dead param for basic transformer
    ff_dim: int = 512