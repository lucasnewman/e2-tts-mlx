import mlx.core as mx
import numpy as np

from huggingface_hub import hf_hub_download

path = hf_hub_download("lucasnewman/vocos-mel-24khz", "model.safetensors")
filterbank = mx.load(path)["feature_extractor.mel_spec.mel_scale.fb"].moveaxis(0, 1)
np.savez_compressed("assets/mel_filters.npz", mel_100=filterbank)
