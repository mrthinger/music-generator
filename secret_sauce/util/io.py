import torchaudio


def get_duration_sec(song_path: str) -> float:
    info = torchaudio.info(song_path)

    return info.num_frames / info.sample_rate
