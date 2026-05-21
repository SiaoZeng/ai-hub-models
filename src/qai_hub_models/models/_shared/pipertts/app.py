# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
import atexit
import functools
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import piper_phonemize
import soundfile as sf
import torch
from qai_hub.client import DatasetEntries
from torch import Tensor
from torch.utils.data import DataLoader

from qai_hub_models.datasets import DatasetSplit, instantiate_dataset
from qai_hub_models.models._shared.pipertts.model import (
    DEC_SEQ_OVERLAP,
    DEFAULT_LENGTH_SCALE,
    DEFAULT_NOISE_SCALE,
    DEFAULT_NOISE_SCALE_W,
    ITALIAN_NOISE_SCALE,
    MAX_DEC_SEQ_LEN,
    MAX_SEQ_LEN,
    SAMPLE_RATE,
    SDP,
    UPSAMPLE_FACTOR,
    UPSAMPLED_MAX_SEQ_LEN,
    Decoder,
    Encoder,
    Flow,
    PiperTTS,
)
from qai_hub_models.models._shared.voiceai_tts.app_utils import (
    ByT5Tokenizer,
    calibrate_charsiu_decoder,
    calibrate_charsiu_encoder,
    generate_path,
)
from qai_hub_models.models._shared.voiceai_tts.language import TTSLanguage
from qai_hub_models.utils.base_app import CollectionAppProtocol
from qai_hub_models.utils.evaluate import sample_dataset
from qai_hub_models.utils.input_spec import InputSpec, get_batch_size
from qai_hub_models.utils.qai_hub_helpers import make_hub_dataset_entries

DEFAULT_TEXTS = {
    TTSLanguage.ITALIAN: "Mi fa piacere che lei sia venuto, spero che resteremo amici.",
    TTSLanguage.GERMAN: "Es wäre besser, wenn du zuerst mit deinem Chef sprichst, bevor du eine Entscheidung triffst.",
    TTSLanguage.ENGLISH: "Effective teamwork relies on clear communication, mutual respect, and shared goals. When team members collaborate openly and support one another, productivity increases and innovative solutions emerge.",
    TTSLanguage.CHINESE: "中文是中国的语言文字。特指汉族的语言文字, 即汉语和汉字",
}
LANGUAGE_MAP_ph = {
    TTSLanguage.ENGLISH: "en-us",
    TTSLanguage.ITALIAN: "it",
    TTSLanguage.GERMAN: "de",
    TTSLanguage.CHINESE: "cmn",
}  # for phonemize


def noise_scale_for_language(language: TTSLanguage) -> float:
    return (
        ITALIAN_NOISE_SCALE if language == TTSLanguage.ITALIAN else DEFAULT_NOISE_SCALE
    )


def run_encoder_sdp(
    encoder: Encoder,
    sdp: SDP,
    phoneme_ids: list[int],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    x, x_lengths = prepare_input(phoneme_ids)
    x_encoded, m_p, logs_p, x_mask = encoder(x, x_lengths)
    y_lengths, w_ceil = sdp(
        x_encoded,
        x_mask,
        torch.tensor([DEFAULT_LENGTH_SCALE], dtype=torch.float32),
        torch.tensor([DEFAULT_NOISE_SCALE_W], dtype=torch.float32),
    )
    y_mask = torch.unsqueeze(
        torch.arange(UPSAMPLED_MAX_SEQ_LEN) < y_lengths.unsqueeze(dim=-1), dim=1
    ).to(torch.float32)
    attn_mask = x_mask.unsqueeze(dim=2) * y_mask.unsqueeze(dim=-1)
    attn = generate_path(w_ceil, attn_mask)
    attn_squeezed = attn.squeeze(1).to(torch.float32)
    return m_p, logs_p, y_mask, attn_squeezed, y_lengths


def run_flow(
    flow: Flow,
    m_p: Tensor,
    logs_p: Tensor,
    y_mask: Tensor,
    attn_squeezed: Tensor,
    noise_scale: float,
) -> Tensor:
    m_p = m_p.to(torch.float32)
    logs_p = logs_p.to(torch.float32)
    noise_scale_pt = torch.tensor([noise_scale], dtype=torch.float32)
    return flow(m_p, logs_p, y_mask, attn_squeezed, noise_scale_pt)


def decode_chunks(
    decoder: Decoder,
    z: Tensor,
    y_lengths: Tensor,
) -> Tensor:
    z_buf = torch.zeros(
        [z.shape[0], z.shape[1], MAX_DEC_SEQ_LEN + 2 * DEC_SEQ_OVERLAP],
        dtype=torch.float32,
    )
    z_buf[:, :, : (MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP)] = z[
        :, :, : (MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP)
    ]
    audio_chunk = decoder(z_buf)
    audio = audio_chunk.squeeze()[: MAX_DEC_SEQ_LEN * UPSAMPLE_FACTOR]
    total_dec_seq_len = MAX_DEC_SEQ_LEN
    while total_dec_seq_len < min(
        int(y_lengths[0]), z.shape[2] - MAX_DEC_SEQ_LEN - DEC_SEQ_OVERLAP
    ):
        z_buf = z[
            :,
            :,
            total_dec_seq_len - DEC_SEQ_OVERLAP : total_dec_seq_len
            + MAX_DEC_SEQ_LEN
            + DEC_SEQ_OVERLAP,
        ]
        audio_chunk = decoder(z_buf)
        audio_chunk = audio_chunk.squeeze()[
            DEC_SEQ_OVERLAP * UPSAMPLE_FACTOR : (MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP)
            * UPSAMPLE_FACTOR
        ]
        audio = torch.cat([audio, audio_chunk])
        total_dec_seq_len += MAX_DEC_SEQ_LEN
    return audio


@functools.cache
def _get_espeak_ng_datapath() -> Path:
    """Resolve the espeak-ng data path, symlinking to a shorter path if needed.

    The C++ library in piper_phonemize latches onto the first data path it
    receives, so the result is cached for the lifetime of the process.
    """
    espeak_data_path = Path(
        os.environ.get("ESPEAK_DATA_PATH", piper_phonemize._DIR)
    ).resolve()

    # The envvar ESPEAK_DATA_PATH is supposed to be set to the PARENT directory
    # of espeak-ng-data, rather than the data dir itself.
    # This is confusing, so we check for both cases.
    espeak_test_path = espeak_data_path / "espeak-ng-data"
    espeak_data_path = (
        espeak_test_path if espeak_test_path.exists() else espeak_data_path
    )

    phontab_path = espeak_data_path / "phontab"
    if not phontab_path.exists():
        raise ValueError(
            f"Invalid espeak_data_path: {phontab_path} does not exist. Is the python package `piper_phonemize` installed correctly? If envvar ESPEAK_DATA_PATH is set, double check its value."
        )

    # 160 is the max path size for the data. It's a hardcoded buffer size.
    # See https://github.com/espeak-ng/espeak-ng/issues/2182
    if len(str(espeak_data_path)) <= 160:
        return espeak_data_path

    if sys.platform == "win32":
        raise ValueError(
            f"The path to the espeak data files is too long: {espeak_data_path}. Copy or symlink it elsewhere and set envvar ESPEAK_DATA_PATH to the directory that contains espeak-ng-data."
        )

    # Symlink a shorter path to the correct data path, and use the shorter
    # path with the CPP lib. The underlying C++ library internally caches the
    # data path on first use and fails if called again with a different path,
    # so this directory must exist for the lifetime of the process.
    #
    # We can't use a context manager because the tmpdir local would be garbage-collected
    # (and deleted on disk) after this function returns. atexit keeps it alive.
    tmpdir = tempfile.TemporaryDirectory()
    atexit.register(tmpdir.cleanup)
    tmp_espeak_data_path = Path(tmpdir.name) / "espeak-ng-data"
    os.symlink(espeak_data_path, tmp_espeak_data_path)
    return tmp_espeak_data_path


@functools.cache
def phonemize_text(text: str, language: str = "en-us") -> list[int]:
    """
    Parameters
    ----------
    text
        text that will be phonemized
    language
        language of text

    Returns
    -------
    list[int]
    """
    datapath = _get_espeak_ng_datapath()
    phonemes = piper_phonemize.phonemize_espeak(text, language, datapath)

    flat_phonemes = []
    for phoneme in phonemes:
        flat_phonemes.extend(phoneme)
        if flat_phonemes and flat_phonemes[-1] not in (".", "?", "!"):
            flat_phonemes.append(".")
    return piper_phonemize.phoneme_ids_espeak(flat_phonemes)


def prepare_input(phoneme_ids: list[int]) -> tuple[Tensor, Tensor]:
    """
    Parameters
    ----------
    phoneme_ids
        the phoneme of text

    Returns
    -------
    x : Tensor
        phoneme_ids after padding or truncating.
    x_lengths : Tensor
        actual length of x.
    """
    actual_length = min(len(phoneme_ids), MAX_SEQ_LEN)
    if len(phoneme_ids) > MAX_SEQ_LEN:
        phoneme_ids = phoneme_ids[:MAX_SEQ_LEN]
    elif len(phoneme_ids) < MAX_SEQ_LEN:
        phoneme_ids = phoneme_ids + [0] * (MAX_SEQ_LEN - len(phoneme_ids))

    # x, x_lengths
    return torch.tensor([phoneme_ids], dtype=torch.int32), torch.tensor(
        [actual_length], dtype=torch.int32
    )


class PiperTTSApp(CollectionAppProtocol):
    def __init__(
        self,
        encoder: Encoder,
        sdp: SDP,
        flow: Flow,
        decoder: Decoder,
        language: TTSLanguage,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.sdp = sdp
        self.flow = flow
        self.decoder = decoder
        self.language = language

    def predict(self, text: str) -> str:
        """
        Parameters
        ----------
        text
            the text needed to synthesized into audio

        Returns
        -------
        output_path : str
            Synthesized audio path.
        """
        output_path = f"piper-audio_{self.language}.wav"
        noise_scale = noise_scale_for_language(self.language)

        phoneme_ids = phonemize_text(text, LANGUAGE_MAP_ph[self.language])
        if len(phoneme_ids) > MAX_SEQ_LEN:
            print(
                f"Input has {len(phoneme_ids)} phonemes, truncating to {MAX_SEQ_LEN}, text={text}"
            )

        m_p, logs_p, y_mask, attn_squeezed, y_lengths = run_encoder_sdp(
            self.encoder, self.sdp, phoneme_ids
        )
        z = run_flow(self.flow, m_p, logs_p, y_mask, attn_squeezed, noise_scale)
        audio = decode_chunks(self.decoder, z, y_lengths)

        audio_np = audio.squeeze().detach().numpy()
        audio_np = audio_np[: int(y_lengths[0]) * UPSAMPLE_FACTOR]
        sf.write(output_path, audio_np, SAMPLE_RATE)
        return output_path

    @classmethod
    def get_calibration_data(
        cls,
        collection_model: PiperTTS,
        component_name: str,
        input_specs: dict[str, InputSpec] | None = None,
        num_samples: int | None = None,
    ) -> DatasetEntries:
        assert hasattr(collection_model, "get_language") and callable(
            collection_model.get_language
        )
        language_ = collection_model.get_language()
        model = collection_model.components[component_name]
        input_spec = (
            input_specs[component_name] if input_specs else model.get_input_spec()
        )
        encoder_fpm = collection_model.encoder
        sdp_fpm = collection_model.sdp
        flow_fpm = collection_model.flow
        T5Encoder = collection_model.charsiu_encoder
        T5Decoder = collection_model.charsiu_decoder
        batch_size = get_batch_size(input_spec) or 1
        assert batch_size == 1, f"Batch size must be 1, found {batch_size}"

        tokenizer = ByT5Tokenizer()
        noise_scale = noise_scale_for_language(language_)

        calibration_dataset_cls = collection_model.get_calibration_dataset_cls()
        assert calibration_dataset_cls is not None
        dataset = instantiate_dataset(
            calibration_dataset_cls,
            DatasetSplit.TRAIN,
            lang=language_,
        )
        num_samples = num_samples or dataset.default_samples_per_job()
        num_samples = (int(num_samples) // batch_size) * batch_size
        if component_name == "sdp":
            speed = 1.0 / model.scale  # type: ignore[operator, unused-ignore]
            print(
                f"\nLoading \033[38;5;206m{num_samples}\033[0m calibration samples for \033[38;5;206m{language_} {component_name}\033[0m component. speed_adjustment = \033[38;5;206m{speed:.3f}\033[0m"
            )
        else:
            print(
                f"\nLoading \033[38;5;206m{num_samples}\033[0m calibration samples for \033[38;5;206m{language_} {component_name}\033[0m component."
            )
        torch_dataset = sample_dataset(dataset, num_samples)
        dataloader = DataLoader(torch_dataset, batch_size=batch_size)

        inputs: list[list[torch.Tensor | np.ndarray]] = [[] for _ in input_spec]
        for text, _ in dataloader:
            if isinstance(text, tuple | list):
                text = text[0]  # batch_size is 1
            if component_name == "encoder":
                phoneme_ids = phonemize_text(text, LANGUAGE_MAP_ph[language_].lower())
                x, x_lengths = prepare_input(phoneme_ids)
                inputs[0].append(x)
                inputs[1].append(x_lengths)

            elif component_name == "sdp":
                phoneme_ids = phonemize_text(text, LANGUAGE_MAP_ph[language_].lower())
                x, x_lengths = prepare_input(phoneme_ids)
                x_encoded, m_p, logs_p, x_mask = encoder_fpm(x, x_lengths)

                inputs[0].append(x_encoded)
                inputs[1].append(x_mask)
                inputs[2].append(
                    torch.tensor((DEFAULT_LENGTH_SCALE,), dtype=torch.float32)
                )
                inputs[3].append(
                    torch.tensor((DEFAULT_NOISE_SCALE_W,), dtype=torch.float32)
                )
            elif component_name == "flow":
                phoneme_ids = phonemize_text(text, LANGUAGE_MAP_ph[language_].lower())
                m_p, logs_p, y_mask, attn_squeezed, y_lengths = run_encoder_sdp(
                    encoder_fpm, sdp_fpm, phoneme_ids
                )
                inputs[0].append(m_p.to(torch.float32))
                inputs[1].append(logs_p.to(torch.float32))
                inputs[2].append(y_mask)
                inputs[3].append(attn_squeezed)
                inputs[4].append(torch.tensor([noise_scale], dtype=torch.float32))

            elif component_name == "decoder":
                phoneme_ids = phonemize_text(text, LANGUAGE_MAP_ph[language_].lower())
                m_p, logs_p, y_mask, attn_squeezed, y_lengths = run_encoder_sdp(
                    encoder_fpm, sdp_fpm, phoneme_ids
                )
                z = run_flow(flow_fpm, m_p, logs_p, y_mask, attn_squeezed, noise_scale)

                z_buf = torch.zeros(
                    [z.shape[0], z.shape[1], MAX_DEC_SEQ_LEN + 2 * DEC_SEQ_OVERLAP],
                    dtype=torch.float32,
                )
                z_buf[:, :, : (MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP)] = z[
                    :, :, : (MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP)
                ]
                inputs[0].append(z_buf)

                total_dec_seq_len = MAX_DEC_SEQ_LEN
                while total_dec_seq_len < min(
                    int(y_lengths[0]), z.shape[2] - MAX_DEC_SEQ_LEN - DEC_SEQ_OVERLAP
                ):
                    z_buf = z[
                        :,
                        :,
                        total_dec_seq_len - DEC_SEQ_OVERLAP : total_dec_seq_len
                        + MAX_DEC_SEQ_LEN
                        + DEC_SEQ_OVERLAP,
                    ]
                    inputs[0].append(z_buf)
                    total_dec_seq_len += MAX_DEC_SEQ_LEN

            elif component_name == "charsiu_encoder":
                calibrate_charsiu_encoder(text, tokenizer, inputs)

            elif component_name == "charsiu_decoder":
                calibrate_charsiu_decoder(text, tokenizer, T5Encoder, T5Decoder, inputs)

            else:
                raise NotImplementedError(component_name)

        return make_hub_dataset_entries(tuple(inputs), list(input_spec.keys()))
