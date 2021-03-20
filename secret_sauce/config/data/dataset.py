from dataclasses import dataclass


@dataclass
class SongsDatasetConfig:
    audio_len_seconds: float = 3
    sample_rate: int = 22000