import torchaudio

# impl for torchaudio 0.8.0+
# def get_duration_sec(song_path: str) -> float:
#     info = torchaudio.info(song_path)
#     return info.num_frames / info.sample_rate


def get_duration_sec(song_path: str) -> float:
    sample_info, encoding_info = torchaudio.info(song_path)

    return sample_info.length / sample_info.rate