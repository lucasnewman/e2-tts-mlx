# This sample will download the "dev-clean" split of the LibriTTS dataset
# and train the model for 100k steps.

import mlx.core as mx
from mlx.utils import tree_flatten, tree_map

from e2_tts_mlx.model import E2TTS
from e2_tts_mlx.trainer import E2Trainer
from e2_tts_mlx.data import load_libritts_r

e2tts = E2TTS(
    tokenizer="phoneme_en",
    cond_drop_prob=0.25,
    frac_lengths_mask=(0.7, 0.9),
    transformer=dict(
        dim=384,
        depth=16,
        heads=8,
        text_depth=8,
        text_heads=8,
        max_seq_len=1024,
        dropout=0.1,
    ),
)

# cast parameters to float16
e2tts.update(tree_map(lambda p: p.astype(mx.float16), e2tts.parameters()))

mx.eval(e2tts.parameters())

num_trainable_params = sum(
    [p[1].size for p in tree_flatten(e2tts.trainable_parameters())]
)
print(f"Using {num_trainable_params:,} trainable parameters.")

batch_size = 4  # adjust based on available memory
max_duration = 10
dataset = load_libritts_r(split="dev-clean", max_duration=max_duration)

trainer = E2Trainer(model=e2tts, num_warmup_steps=5_000, max_grad_norm=1)
trainer.train(
    train_dataset=dataset,
    learning_rate=7.5e-5,
    log_every=10,
    plot_every=1000,
    total_steps=100_000,
    batch_size=batch_size,
)
