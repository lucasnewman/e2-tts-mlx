from __future__ import annotations
from collections import namedtuple
from functools import lru_cache, partial
import math
import os
from typing import Literal, Callable, Union

import mlx.core as mx
import mlx.nn as nn

import numpy as np

from einops.array_api import rearrange, reduce, repeat, pack
import einx

from vocos_mlx import Vocos

from g2p_en import G2p

E2TTSReturn = namedtuple("E2TTS", ["loss", "cond", "pred_flow", "pred_data", "flow"])


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def divisible_by(num, den):
    return (num % den) == 0


def lens_to_mask(
    t: mx.array,
    length: int | None = None,
) -> mx.array:  # Bool['b n']
    if not exists(length):
        length = t.max()

    seq = mx.arange(length)
    return einx.less("n, b -> b n", seq, t)


def mask_from_start_end_indices(
    seq_len: mx.array,
    start: mx.array,
    end: mx.array,
    max_length: int | None = None,
):
    max_seq_len = default(max_length, seq_len.max().item())
    seq = mx.arange(max_seq_len).astype(mx.int32)
    return einx.greater_equal("n, b -> b n", seq, start) & einx.less(
        "n, b -> b n", seq, end
    )


def mask_from_frac_lengths(
    seq_len: mx.array,
    frac_lengths: mx.array,
    max_length: int | None = None,
):
    lengths = (frac_lengths * seq_len).astype(mx.int32)
    max_start = seq_len - lengths

    rand = mx.random.uniform(0, 1, frac_lengths.shape)

    start = mx.maximum((max_start * rand).astype(mx.int32), 0)
    end = start + lengths

    out = mask_from_start_end_indices(seq_len, start, end, max_length)

    if exists(max_length):
        out = pad_to_length(out, max_length)

    return out


def maybe_masked_mean(t: mx.array, mask: mx.array | None = None) -> mx.array:
    if not exists(mask):
        return t.mean(dim=1)

    t = einx.where("b n, b n d, -> b n d", mask, t, 0.0)
    num = reduce(t, "b n d -> b d", "sum")
    den = reduce(mask.astype(mx.int32), "b n -> b", "sum")

    return einx.divide("b d, b -> b d", num, mx.maximum(den, 1))


def pad_to_length(t: mx.array, length: int, value=None):
    ndim = t.ndim
    seq_len = t.shape[-1]
    if length > seq_len:
        if ndim == 1:
            t = mx.pad(t, [(0, length - seq_len)], constant_values=value)
        elif ndim == 2:
            t = mx.pad(t, [(0, 0), (0, length - seq_len)], constant_values=value)
        else:
            raise ValueError(f"Unsupported padding dims: {ndim}")
    return t[..., :length]


def pad_sequence(t: mx.array, padding_value=0):
    max_len = max([i.shape[-1] for i in t])
    t = mx.array([pad_to_length(i, max_len, padding_value) for i in t])
    return t


# simple utf-8 tokenizer, since paper went character based


def list_str_to_tensor(text: list[str], padding_value=-1) -> mx.array:  # Int['b nt']:
    list_tensors = [mx.array([*bytes(t, "UTF-8")]) for t in text]
    padded_tensor = pad_sequence(list_tensors, padding_value=-1)
    return padded_tensor


# simple english phoneme-based tokenizer


def get_g2p_en_encode():
    g2p = G2p()

    phoneme_to_index = g2p.p2idx
    num_phonemes = len(phoneme_to_index)

    extended_chars = [
        " ",
        ",",
        ".",
        "-",
        "!",
        "?",
        "'",
        '"',
        "...",
        "..",
        ". .",
        ". . .",
        ". . . .",
        ". . . . .",
        ". ...",
        "... .",
        ".. ..",
    ]
    num_extended_chars = len(extended_chars)

    extended_chars_dict = {p: (num_phonemes + i) for i, p in enumerate(extended_chars)}
    phoneme_to_index = {**phoneme_to_index, **extended_chars_dict}

    def encode(text: list[str], padding_value=-1) -> mx.array:
        phonemes = [g2p(t) for t in text]
        list_tensors = [
            mx.array([phoneme_to_index[p] for p in one_phoneme], dtype=mx.int32)
            for one_phoneme in phonemes
        ]
        padded_tensor = pad_sequence(list_tensors, padding_value=-1)
        return padded_tensor

    return encode, (num_phonemes + num_extended_chars)


# mel spectrogram


@lru_cache(maxsize=None)
def mel_filters(n_mels: int) -> mx.array:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Saved using extract_filterbank.py
    """
    assert n_mels in {100}, f"Unsupported n_mels: {n_mels}"

    filename = os.path.join("assets", "mel_filters.npz")
    return mx.load(filename, format="npz")[f"mel_{n_mels}"]


@lru_cache(maxsize=None)
def hanning(size):
    return mx.array(np.hanning(size + 1)[:-1])


def stft(x, window, nperseg=256, noverlap=None, nfft=None, pad_mode="constant"):
    if nfft is None:
        nfft = nperseg
    if noverlap is None:
        noverlap = nfft // 4

    def _pad(x, padding, pad_mode="constant"):
        if pad_mode == "constant":
            return mx.pad(x, [(padding, padding)])
        elif pad_mode == "reflect":
            prefix = x[1 : padding + 1][::-1]
            suffix = x[-(padding + 1) : -1][::-1]
            return mx.concatenate([prefix, x, suffix])
        else:
            raise ValueError(f"Invalid pad_mode {pad_mode}")

    padding = nperseg // 2
    x = _pad(x, padding, pad_mode)

    strides = [noverlap, 1]
    t = (x.size - nperseg + noverlap) // noverlap
    shape = [t, nfft]
    x = mx.as_strided(x, shape=shape, strides=strides)
    return mx.fft.rfft(x * window)


def log_mel_spectrogram(
    audio: Union[mx.array, np.ndarray],
    n_mels: int = 100,
    n_fft: int = 1024,
    hop_length: int = 256,
    padding: int = 0,
    filterbank: mx.array | None = None,
):
    if not isinstance(audio, mx.array):
        audio = mx.array(audio)

    if padding > 0:
        audio = mx.pad(audio, (0, padding))

    freqs = stft(audio, hanning(n_fft), nperseg=n_fft, noverlap=hop_length)
    magnitudes = freqs[:-1, :].abs()
    filters = filterbank if filterbank is not None else mel_filters(n_mels)
    mel_spec = magnitudes @ filters.T
    log_spec = mx.maximum(mel_spec, 1e-5).log()
    return mx.expand_dims(log_spec, axis=0)


class MelSpec(nn.Module):
    def __init__(
        self,
        sample_rate=24_000,
        n_fft=1024,
        hop_length=256,
        n_mels=100,
        padding="center",
        filterbank: mx.array | None = None,
    ):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.sample_rate = sample_rate
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.filterbank = filterbank

    def __call__(self, audio: mx.array, **kwargs) -> mx.array:
        return log_mel_spectrogram(
            audio,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            padding=0,
            filterbank=self.filterbank,
        )


# adaln zero from DiT paper


class AdaLNZero(nn.Module):
    def __init__(self, dim, dim_condition=None, init_bias_value=-2.0):
        super().__init__()
        dim_condition = default(dim_condition, dim)
        self.to_gamma = nn.Linear(dim_condition, dim)

        self.to_gamma.weight = mx.zeros_like(self.to_gamma.weight)
        self.to_gamma.bias = nn.init.constant(init_bias_value)(
            mx.zeros_like(self.to_gamma.bias)
        )

    def __call__(self, x, *, condition):
        if condition.ndim == 2:
            condition = rearrange(condition, "b d -> b 1 d")

        gamma = nn.sigmoid(self.to_gamma(condition))
        return x * gamma


# random projection fourier embedding


class RandomFourierEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        self.weights = mx.random.normal((dim // 2,))

    def __call__(self, x):
        freqs = einx.multiply("i, j -> i j", x, self.weights) * 2 * math.pi
        fourier_embed, _ = pack((x, freqs.sin(), freqs.cos()), "b *")
        return fourier_embed


# character embedding


class CharacterEmbed(nn.Module):
    def __init__(self, dim, num_embeds=256, max_seq_len=512):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # will just use 0 as the 'filler token'
        self.embed = nn.Embedding(num_embeds + 1, dim)
        self.abs_pos_embed = nn.Embedding(max_seq_len, dim)

    def __call__(
        self,
        text: mx.array,  # Int['b nt']
        max_seq_len: int,
        **kwargs,
    ) -> mx.array:  # Float['b n d']
        if max_seq_len > self.max_seq_len:
            raise ValueError(
                f"max_seq_len ({max_seq_len}) exceeds the set maximum sequence length ({self.max_seq_len})"
            )

        text = text + 1  # shift all other token ids up by 1 and use 0 as filler token

        # just curtail if character tokens are more than the mel spec tokens, one of the edge cases the paper did not address
        text = text[:, :max_seq_len]

        text = pad_to_length(text, max_seq_len, value=0)

        embeddings = self.embed(text)
        pos_emb = self.abs_pos_embed(mx.arange(max_seq_len))
        embeddings = embeddings + pos_emb

        return embeddings


# text audio cross conditioning in multistream setup


class TextAudioCrossCondition(nn.Module):
    def __init__(
        self,
        dim,
        dim_text,
        cond_audio_to_text=True,
    ):
        super().__init__()
        self.text_to_audio = nn.Linear(dim_text + dim, dim, bias=False)
        self.text_to_audio.weight = mx.zeros_like(self.text_to_audio.weight)

        self.cond_audio_to_text = cond_audio_to_text

        if cond_audio_to_text:
            self.audio_to_text = nn.Linear(dim + dim_text, dim_text, bias=False)
            self.audio_to_text.weight = mx.zeros_like(self.audio_to_text.weight)

    def __call__(
        self,
        audio: mx.array,  # Float['b n d'],
        text: mx.array,  # Float['b n dt']
    ):
        audio_text, _ = pack((audio, text), "b n *")

        text_cond = self.text_to_audio(audio_text)
        audio_cond = self.audio_to_text(audio_text) if self.cond_audio_to_text else 0.0

        return audio + text_cond, text + audio_cond


# attention and transformer backbone
# for use in both e2tts as well as duration module


class Identity(nn.Module):
    def __call__(self, x):
        return x


class AdaptiveRMSNorm(nn.Module):
    def __init__(self, dim, dim_condition=None):
        super().__init__()
        self.scale = dim**0.5
        dim_condition = default(dim_condition, dim)

        self.to_gamma = nn.Linear(dim_condition, dim, bias=False)
        self.to_gamma.weight = mx.zeros_like(self.to_gamma.weight)

    def __call__(self, x, *, condition):
        if condition.ndim == 2:
            condition = rearrange(condition, "b d -> b 1 d")

        weight = mx.ones(x.shape[-1])
        normed = mx.fast.rms_norm(x, weight, eps=1e-5)
        gamma = self.to_gamma(condition)
        return normed * self.scale * (gamma + 1.0)


def get_attn_mask(mask: mx.array | None) -> mx.array | None:
    if exists(mask):
        mask = rearrange(mask, "b n -> b 1 1 n")
        mask = mx.repeat(mask, repeats=mask.shape[-1], axis=2)
    return mask


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        kv_heads: int | None = None,
        rotary_pos_emb: mx.array | None = None,
    ):
        super().__init__()
        self.n_heads: int = heads
        self.n_kv_heads: int = default(kv_heads, heads)

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = dim_head**-0.5

        self.wq = nn.Linear(dim, self.n_heads * dim_head, bias=False)
        self.wk = nn.Linear(dim, self.n_kv_heads * dim_head, bias=False)
        self.wv = nn.Linear(dim, self.n_kv_heads * dim_head, bias=False)
        self.wo = nn.Linear(self.n_heads * dim_head, dim, bias=False)
        self.rope = nn.RoPE(dim_head, traditional=True)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        if exists(mask):
            mask = get_attn_mask(mask)

        B, L, D = x.shape

        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        queries = self.rope(queries)
        keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        mult=4,
        dropout=0.0,
        no_bias=False,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        activation = nn.GELU()
        self.ff = nn.Sequential(
            nn.Linear(dim, inner_dim, bias=not no_bias),
            activation,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias=not no_bias),
        )

    def __call__(self, x):
        return self.ff(x)


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_text=None,  # will default to half of audio dimension
        depth=8,
        heads=8,
        dim_head=64,
        ff_mult=4,
        text_depth=None,
        text_heads=None,
        text_dim_head=None,
        text_ff_mult=None,
        cond_on_time=True,
        abs_pos_emb=True,
        max_seq_len=4096,
        dropout=0.1,
        ff_kwargs: dict = dict(),
    ):
        super().__init__()
        assert divisible_by(depth, 2), "depth needs to be even"

        # absolute positional embedding

        self.max_seq_len = max_seq_len
        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if abs_pos_emb else None

        self.dim = dim

        dim_text = default(dim_text, dim // 2)
        self.dim_text = dim_text

        text_heads = default(text_heads, heads)
        text_dim_head = default(text_dim_head, dim_head)
        text_ff_mult = default(text_ff_mult, ff_mult)
        text_depth = default(text_depth, depth)

        assert (
            1 <= text_depth <= depth
        ), "must have at least 1 layer of text conditioning, but less than total number of speech layers"

        self.depth = depth
        self.layers = []

        # rotary embedding

        self.rotary_emb = nn.RoPE(dim_head)
        self.text_rotary_emb = nn.RoPE(dim_head)

        # time conditioning
        # will use adaptive rmsnorm

        self.cond_on_time = cond_on_time
        rmsnorm_klass = nn.RMSNorm if not cond_on_time else AdaptiveRMSNorm
        postbranch_klass = Identity if not cond_on_time else partial(AdaLNZero, dim=dim)

        self.time_cond_mlp = Identity()

        if cond_on_time:
            self.time_cond_mlp = nn.Sequential(
                RandomFourierEmbed(dim), nn.Linear(dim + 1, dim), nn.SiLU()
            )

        for ind in range(depth):
            is_later_half = ind >= (depth // 2)
            has_text = ind < text_depth

            # speech related

            attn_norm = rmsnorm_klass(dim)
            attn = nn.MultiHeadAttention(dims=dim, num_heads=heads)
            attn_adaln_zero = postbranch_klass()

            ff_norm = rmsnorm_klass(dim)
            ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, **ff_kwargs)
            ff_adaln_zero = postbranch_klass()

            skip_proj = nn.Linear(dim * 2, dim, bias=False) if is_later_half else None

            speech_modules = [
                skip_proj,
                attn_norm,
                attn,
                attn_adaln_zero,
                ff_norm,
                ff,
                ff_adaln_zero,
            ]

            text_modules = None

            if has_text:
                # text related

                text_attn_norm = nn.RMSNorm(dim_text)
                text_attn = nn.MultiHeadAttention(dims=dim_text, num_heads=text_heads)

                text_ff_norm = nn.RMSNorm(dim_text)
                text_ff = FeedForward(
                    dim=dim_text,
                    mult=text_ff_mult,
                    dropout=dropout,
                    **ff_kwargs,
                )

                # cross condition

                is_last = ind == (text_depth - 1)

                cross_condition = TextAudioCrossCondition(
                    dim=dim, dim_text=dim_text, cond_audio_to_text=not is_last
                )

                text_modules = [
                    text_attn_norm,
                    text_attn,
                    text_ff_norm,
                    text_ff,
                    cross_condition,
                ]

            self.layers.append([speech_modules, text_modules])

        self.final_norm = nn.RMSNorm(dim)

    def __call__(
        self,
        x: mx.array,
        times: mx.array | None = None,
        mask: mx.array | None = None,
        text_embed: mx.array | None = None,
    ):
        batch, seq_len = x.shape[:2]

        assert not (
            exists(times) ^ self.cond_on_time
        ), "`times` must be passed in if `cond_on_time` is set to `True` and vice versa"

        # handle absolute positions if needed

        if exists(self.abs_pos_emb):
            assert (
                seq_len <= self.max_seq_len
            ), f"{seq_len} exceeds the set `max_seq_len` ({self.max_seq_len}) on Transformer"
            seq = mx.arange(seq_len)
            x = x + self.abs_pos_emb(seq)

        # handle adaptive rmsnorm kwargs

        norm_kwargs = dict()

        if exists(times):
            if times.ndim == 0:
                times = repeat(times, " -> b", b=batch)

            times = self.time_cond_mlp(times)
            norm_kwargs.update(condition=times)

        # skip connection related stuff

        skips = []

        # go through the layers

        # print(f"depth: {self.depth}, x: {x}")

        for ind, (speech_modules, text_modules) in enumerate(self.layers):
            layer = ind + 1

            (
                maybe_skip_proj,
                attn_norm,
                attn,
                maybe_attn_adaln_zero,
                ff_norm,
                ff,
                maybe_ff_adaln_zero,
            ) = speech_modules

            # smaller text transformer

            if exists(text_embed) and exists(text_modules):
                (text_attn_norm, text_attn, text_ff_norm, text_ff, cross_condition) = (
                    text_modules
                )

                text_attn_embed = text_attn_norm(text_embed)
                text_attn_embed = text_attn(
                    text_attn_embed,
                    text_attn_embed,
                    text_attn_embed,
                    mask=get_attn_mask(mask),
                )

                text_embed = text_attn_embed + text_embed
                text_ff_out = text_ff(text_ff_norm(text_embed))
                text_embed = text_ff_out + text_embed

                x, text_embed = cross_condition(x, text_embed)

            # skip connection logic

            is_first_half = layer <= (self.depth // 2)
            is_later_half = not is_first_half

            if is_first_half:
                skips.append(x)

            if is_later_half:
                skip = skips.pop()
                x = mx.concatenate((x, skip), axis=-1)
                x = maybe_skip_proj(x)

            # attention and feedforward blocks

            x = attn_norm(x, **norm_kwargs)

            attn_out = attn(x, x, x, mask=get_attn_mask(mask))

            x = x + maybe_attn_adaln_zero(attn_out, **norm_kwargs)

            ff_out = ff(ff_norm(x, **norm_kwargs))

            x = x + maybe_ff_adaln_zero(ff_out, **norm_kwargs)

        assert len(skips) == 0

        return self.final_norm(x)


class Rearrange(nn.Module):
    def __init__(self, pattern: str):
        super().__init__()
        self.pattern = pattern

    def __call__(self, x: mx.array) -> mx.array:
        return rearrange(x, self.pattern)


# main classes


class DurationPredictor(nn.Module):
    def __init__(
        self,
        transformer: dict | Transformer,
        num_channels=None,
        mel_spec_kwargs: dict = dict(),
        char_embed_kwargs: dict = dict(),
        text_num_embeds=None,
        tokenizer: (
            Literal["char_utf8", "phoneme_en"] | Callable[[list[str]]]
        ) = "char_utf8",
    ):
        super().__init__()

        if isinstance(transformer, dict):
            transformer = Transformer(**transformer, cond_on_time=False)

        # mel spec

        self.mel_spec = MelSpec(**mel_spec_kwargs)
        self.num_channels = default(num_channels, self.mel_spec.n_mels)

        self.transformer = transformer

        dim = transformer.dim
        dim_text = transformer.dim_text

        self.dim = dim

        self.proj_in = nn.Linear(self.num_channels, self.dim)

        # tokenizer and text embed

        if callable(tokenizer):
            assert exists(
                text_num_embeds
            ), "`text_num_embeds` must be given if supplying your own tokenizer encode function"
            self.tokenizer = tokenizer
        elif tokenizer == "char_utf8":
            text_num_embeds = 256
            self.tokenizer = list_str_to_tensor
        elif tokenizer == "phoneme_en":
            self.tokenizer, text_num_embeds = get_g2p_en_encode()
        else:
            raise ValueError(f"unknown tokenizer string {tokenizer}")

        self.embed_text = CharacterEmbed(
            dim_text, num_embeds=text_num_embeds, **char_embed_kwargs
        )

        # to prediction

        self.to_pred = nn.Sequential(
            nn.Linear(dim, 1, bias=False), nn.Softplus(), Rearrange("... 1 -> ...")
        )

    def __call__(
        self,
        x: mx.array,
        *,
        text: mx.array | list[str] | None = None,
        lens: mx.array | None = None,
        return_loss=True,
    ):
        # raw wave

        if x.ndim == 2:
            x = self.mel_spec(x)
            x = rearrange(x, "b d n -> b n d")
            assert x.shape[-1] == self.dim

        x = self.proj_in(x)

        batch, seq_len = x.shape[:2]

        # text

        text_embed = None

        if exists(text):
            if isinstance(text, list):
                text = list_str_to_tensor(text)
                assert text.shape[0] == batch

            text_embed = self.embed_text(text, seq_len)

        # handle lengths (duration)

        if not exists(lens):
            lens = mx.full((batch,), seq_len)

        mask = lens_to_mask(lens, length=seq_len)

        # if returning a loss, mask out randomly from an index and have it predict the duration

        if return_loss:
            rand_frac_index = x.new_zeros(batch).uniform_(0, 1)
            rand_index = (rand_frac_index * lens).astype(mx.int32)

            seq = mx.arange(seq_len)
            mask &= einx.less("n, b -> b n", seq, rand_index)

        # attending

        x = self.transformer(
            x,
            mask=mask,
            text_embed=text_embed,
        )

        x = maybe_masked_mean(x, mask)

        pred = self.to_pred(x)

        # return the prediction if not returning loss

        if not return_loss:
            return pred

        # loss

        return nn.losses.mse_loss(pred, lens.float())


class E2TTS(nn.Module):
    def __init__(
        self,
        transformer: dict | Transformer = None,
        duration_predictor: dict | DurationPredictor | None = None,
        cond_drop_prob=0.25,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        char_embed_kwargs: dict = dict(),
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        concat_cond=False,
        text_num_embeds: int | None = None,
        tokenizer: (
            Literal["char_utf8", "phoneme_en"] | Callable[[list[str]]]
        ) = "char_utf8",
        use_vocos=True,
        pretrained_vocos_path="lucasnewman/vocos-mel-24khz",
    ):
        super().__init__()

        if isinstance(transformer, dict):
            transformer = Transformer(**transformer, cond_on_time=True)

        if isinstance(duration_predictor, dict):
            duration_predictor = DurationPredictor(**duration_predictor)

        self.transformer = transformer

        dim = transformer.dim
        dim_text = transformer.dim_text

        self.dim = dim
        self.dim_text = dim_text

        self.frac_lengths_mask = frac_lengths_mask

        self.duration_predictor = duration_predictor

        # mel spec

        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mels)

        self.num_channels = num_channels

        # whether to concat condition and project rather than project both and sum

        self.concat_cond = concat_cond

        if concat_cond:
            self.proj_in = nn.Linear(num_channels * 2, dim)
        else:
            self.proj_in = nn.Linear(num_channels, dim)
            self.cond_proj_in = nn.Linear(num_channels, dim)

        # to prediction

        self.to_pred = nn.Linear(dim, num_channels)

        # tokenizer and text embed

        if callable(tokenizer):
            assert exists(
                text_num_embeds
            ), "`text_num_embeds` must be given if supplying your own tokenizer encode function"
            self.tokenizer = tokenizer
        elif tokenizer == "char_utf8":
            text_num_embeds = 256
            self.tokenizer = list_str_to_tensor
        elif tokenizer == "phoneme_en":
            self.tokenizer, text_num_embeds = get_g2p_en_encode()
        else:
            raise ValueError(f"unknown tokenizer string {tokenizer}")

        self.cond_drop_prob = cond_drop_prob

        # text embedding

        self.embed_text = CharacterEmbed(
            dim_text,
            num_embeds=text_num_embeds,
            max_seq_len=transformer.max_seq_len,
            **char_embed_kwargs,
        )

        # default vocos for mel -> audio

        self.vocos = (
            Vocos.from_pretrained(pretrained_vocos_path).freeze() if use_vocos else None
        )

    def transformer_with_pred_head(
        self,
        x: mx.array,
        cond: mx.array,
        times: mx.array,
        mask: mx.array | None = None,
        text: mx.array | None = None,
        drop_text_cond: bool | None = None,
    ):
        seq_len = x.shape[-2]
        drop_text_cond = default(
            drop_text_cond, self.training and np.random.random() < self.cond_drop_prob
        )

        if self.concat_cond:
            # concat condition, given as using voicebox-like scheme
            x = mx.concatenate((cond, x), dim=-1)

        x = self.proj_in(x)

        if not self.concat_cond:
            # an alternative is to simply sum the condition
            # seems to work fine

            cond = self.cond_proj_in(cond)
            x = x + cond

        # whether to use a text embedding

        text_embed = None
        if exists(text) and not drop_text_cond:
            text_embed = self.embed_text(text, seq_len, mask=mask)

        # attend

        attended = self.transformer(x, times=times, mask=mask, text_embed=text_embed)

        return self.to_pred(attended)

    def cfg_transformer_with_pred_head(
        self,
        *args,
        cfg_strength: float = 1.0,
        **kwargs,
    ):
        pred = self.transformer_with_pred_head(*args, drop_text_cond=False, **kwargs)

        if cfg_strength < 1e-5:
            return pred

        null_pred = self.transformer_with_pred_head(
            *args, drop_text_cond=True, **kwargs
        )

        return pred + (pred - null_pred) * cfg_strength

    def odeint(self, func, y0, t):
        """
        Solves ODE using the midpoint method.

        Parameters:
        - y0: Initial state, an MLX array of any shape.
        - t: Array of time steps, an MLX array.
        """
        ys = [y0]
        y_current = y0

        for i in range(len(t) - 1):
            t_current = t[i]
            dt = t[i + 1] - t_current

            # midpoint approximation
            k1 = func(t_current, y_current)
            mid = y_current + 0.5 * dt * k1

            # compute the next value
            k2 = func(t_current + 0.5 * dt, mid)
            y_next = y_current + dt * k2

            ys.append(y_next)
            y_current = y_next

        return mx.stack(ys)

    def sample(
        self,
        cond: mx.array,
        *,
        text: mx.array | None = None,
        lens: mx.array | None = None,
        duration: mx.array | int | None = None,
        steps=32,
        cfg_strength=1.0,
        max_duration=4096,
        vocoder: Callable[[mx.array]] | None = None,
        return_raw_output: bool | None = None,
    ):
        self.eval()

        # raw wave

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = rearrange(cond, "b d n -> b n d")
            assert cond.shape[-1] == self.num_channels

        batch, cond_seq_len = cond.shape[:2]

        if not exists(lens):
            lens = mx.full((batch,), cond_seq_len, dtype=mx.int32)

        # text

        if isinstance(text, list):
            text = self.tokenizer(text)
            assert text.shape[0] == batch

        if exists(text):
            text_lens = (text != -1).sum(axis=-1)
            lens = mx.maximum(
                text_lens, lens
            )  # make sure lengths are at least those of the text characters

        # duration

        cond_mask = lens_to_mask(lens)

        if exists(duration):
            if isinstance(duration, int):
                duration = mx.full((batch,), duration, dtype=mx.int32)

        elif exists(self.duration_predictor):
            duration = self.duration_predictor(
                cond, text=text, lens=lens, return_loss=False
            ).astype(mx.int32)

        duration = mx.maximum(
            lens + 1, duration
        )  # just add one token so something is generated
        duration = mx.minimum(duration, max_duration)

        assert duration.shape[0] == batch

        max_duration = duration.max().item()

        cond = mx.pad(
            cond, [(0, 0), (0, max_duration - cond_seq_len), (0, 0)], constant_values=0
        )
        cond_mask = mx.pad(
            cond_mask,
            [(0, 0), (0, max_duration - cond_mask.shape[-1])],
            constant_values=False,
        )
        cond_mask = rearrange(cond_mask, "... -> ... 1")

        mask = lens_to_mask(duration)

        # neural ode

        def fn(t, x):
            # at each step, conditioning is fixed

            step_cond = mx.where(cond_mask, cond, mx.zeros_like(cond))

            # predict flow

            return self.cfg_transformer_with_pred_head(
                x, step_cond, times=t, text=text, mask=mask, cfg_strength=cfg_strength
            )

        y0 = mx.random.normal(cond.shape)
        t = mx.linspace(0, 1, steps)

        trajectory = self.odeint(fn, y0, t)
        sampled = trajectory[-1]

        out = sampled

        out = mx.where(cond_mask, cond, out)

        # able to return raw untransformed output, if not using mel rep

        if exists(return_raw_output) and return_raw_output:
            return out

        # take care of transforming mel to audio if `vocoder` is passed in, or if `use_vocos` is turned on

        if exists(vocoder):
            assert not exists(
                self.vocos
            ), "`use_vocos` should not be turned on if you are passing in a custom `vocoder` on sampling"
            out = rearrange(out, "b n d -> b d n")
            out = vocoder(out)

        elif exists(self.vocos):
            audio = []
            for mel, one_mask in zip(out, mask):
                one_out = mx.array(np.array(mel)[one_mask])
                one_out = rearrange(one_out, "n d -> 1 n d")
                one_audio = self.vocos.decode(one_out)
                audio.append(one_audio)

            out = audio

        return out

    def __call__(
        self,
        inp: mx.array,  # mel or raw wave
        *,
        text: mx.array | list[str] | None = None,
        times: mx.array | None = None,
        lens: mx.array | None = None,
    ):
        # handle raw wave

        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = rearrange(inp, "b d n -> b n d")
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype = *inp.shape[:2], inp.dtype

        # handle text as string

        if isinstance(text, list):
            text = self.tokenizer(text)
            assert text.shape[0] == batch

        # lens and mask

        if not exists(lens):
            lens = mx.full((batch,), seq_len)

        mask = lens_to_mask(lens, length=seq_len)

        # get a random span to mask out for training conditionally

        frac_lengths = mx.random.uniform(*self.frac_lengths_mask, (batch,))
        rand_span_mask = mask_from_frac_lengths(lens, frac_lengths, max_length=seq_len)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1

        x1 = inp

        # main conditional flow training logic

        # x0 is gaussian noise

        x0 = mx.random.normal(x1.shape)

        # t is random times

        times = mx.random.uniform(0, 1, (batch,), dtype=dtype)
        t = rearrange(times, "b -> b 1 1")

        # sample xt

        w = (1.0 - ((1.0 - 1e-5) * t)) * x0 + t * x1

        flow = x1 - (1.0 - 1e-5) * x0

        # only predict what is within the random mask span for infilling

        cond = einx.where(
            "b n, b n d, b n d -> b n d", rand_span_mask, mx.zeros_like(x1), x1
        )

        # transformer and prediction head

        pred_flow = self.transformer_with_pred_head(
            w, cond, times=times, text=text, mask=mask
        )
        pred_data = x0 + pred_flow

        # flow matching loss

        loss = nn.losses.mse_loss(pred_flow, flow, reduction="none")

        rand_span_mask = repeat(rand_span_mask, "b n -> b n d", d=self.num_channels)
        masked_loss = mx.where(rand_span_mask, loss, mx.zeros_like(loss))
        loss = mx.sum(masked_loss) / mx.maximum(mx.sum(rand_span_mask), 1e-6)

        return E2TTSReturn(loss, cond, pred_flow, pred_data, flow)
