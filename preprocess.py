from pathlib import Path
from tqdm import tqdm

import mlx.core as mx

from e2_tts_mlx.model import log_mel_spectrogram

import torchaudio


# utilities


def files_with_extensions(root: Path, extensions: list = ["wav"]):
    files = []
    for ext in extensions:
        files.extend(list(root.rglob(f"*.{ext}")))
    files = sorted(files)
    return files


path = Path("...").expanduser()
files = files_with_extensions(path)

for file in tqdm(files):
    mel_file = file.with_suffix(".mel.npy")
    if mel_file.exists():
        continue

    audio, sr = torchaudio.load(file)
    audio = audio.squeeze().numpy()
    mx.savez(mel_file.as_posix(), log_mel_spectrogram(audio))
