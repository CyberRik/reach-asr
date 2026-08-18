"""Tests for the degradation pipeline.

These assert measurable signal properties, not that the code runs. An
augmentation pipeline is uniquely easy to get silently wrong: a mis-scaled noise
mix or a filter that does nothing still produces audio that sounds plausible and
trains without error, and the only symptom is a WER result that means something
other than what you claim it means. Each test below pins one property the WER
comparison actually depends on.
"""

from __future__ import annotations

import math

import pytest
import torch

from reach_asr.telephony import (
    PASSBAND_HIGH_HZ,
    WHISPER_SR,
    DegradationConfig,
    apply_g711,
    band_limit,
    degrade,
    drop_packets,
    mix_noise,
)


def tone(freq_hz: float, seconds: float = 1.0, sr: int = WHISPER_SR) -> torch.Tensor:
    t = torch.arange(int(sr * seconds), dtype=torch.float32) / sr
    return torch.sin(2 * math.pi * freq_hz * t).unsqueeze(0)


def rms(x: torch.Tensor) -> float:
    return float(x.pow(2).mean().sqrt())


def test_band_limit_keeps_passband_and_kills_the_rest() -> None:
    # 1 kHz sits inside the telephone passband; 6 kHz is well outside it.
    kept = band_limit(tone(1000.0), WHISPER_SR)
    killed = band_limit(tone(6000.0), WHISPER_SR)

    assert rms(kept) > 0.5 * rms(tone(1000.0)), "in-band tone should survive"
    assert rms(killed) < 0.1 * rms(tone(6000.0)), "out-of-band tone should be attenuated"


def test_band_limit_removes_the_fricative_band() -> None:
    """The property that makes telephone speech hard: energy above ~3.4 kHz,
    where /s/ and /f/ live, is gone. This is the actual reason a wideband-trained
    model degrades on calls, so it is worth asserting rather than assuming."""
    above = band_limit(tone(PASSBAND_HIGH_HZ * 2), WHISPER_SR)
    assert rms(above) < 0.05


def test_g711_round_trip_is_lossy_but_preserves_the_signal() -> None:
    clean = tone(1000.0)
    coded = apply_g711(clean, WHISPER_SR)

    assert coded.shape == clean.shape, "must return to 16 kHz at the same length"
    assert not torch.allclose(coded, clean, atol=1e-3), "a lossy codec must change the signal"
    # Still recognisably the same tone: correlated, not destroyed.
    correlation = float(
        torch.dot(coded.flatten(), clean.flatten()) / (coded.norm() * clean.norm() + 1e-9)
    )
    assert correlation > 0.7, f"signal should survive the codec, got correlation {correlation}"


@pytest.mark.parametrize("target_snr", [0.0, 5.0, 10.0, 20.0])
def test_mix_noise_hits_the_requested_snr(target_snr: float) -> None:
    """The mix must land on the SNR it was asked for.

    This is the single most important assertion here. Every WER number is
    reported against an SNR, so if the mixing maths is off by a scale factor,
    every result is mislabelled -- and nothing else in the pipeline would fail.
    """
    speech = tone(440.0)
    noise = torch.randn(1, WHISPER_SR)
    generator = torch.Generator().manual_seed(0)

    mixed = mix_noise(speech, noise, target_snr, generator)

    added = mixed - speech
    measured = 10 * math.log10(speech.pow(2).mean() / added.pow(2).mean())
    assert abs(measured - target_snr) < 0.5, f"asked {target_snr} dB, measured {measured:.2f} dB"


def test_mix_noise_tiles_short_noise_without_truncating_speech() -> None:
    speech = tone(440.0, seconds=3.0)
    short_noise = torch.randn(1, WHISPER_SR // 4)  # 0.25 s against 3 s of speech

    mixed = mix_noise(speech, short_noise, 10.0, torch.Generator().manual_seed(0))

    assert mixed.shape == speech.shape


def test_drop_packets_removes_roughly_the_requested_fraction() -> None:
    wave = torch.ones(1, WHISPER_SR * 10)
    out = drop_packets(wave, WHISPER_SR, loss_rate=0.10, packet_ms=20.0, generator=torch.Generator().manual_seed(0))

    zero_fraction = float((out == 0).float().mean())
    assert 0.05 < zero_fraction < 0.16, f"expected ~10% dropped, got {zero_fraction:.3f}"


def test_drop_packets_is_a_noop_at_zero_rate() -> None:
    wave = tone(440.0)
    out = drop_packets(wave, WHISPER_SR, 0.0, 20.0, torch.Generator().manual_seed(0))
    assert torch.equal(out, wave)


def test_degrade_is_deterministic_for_a_given_seed() -> None:
    """Reproducibility is not hygiene here, it is correctness: the baseline and
    the fine-tuned model must be scored on byte-identical audio, or the WER
    difference between them is partly just a different random mix."""
    speech = tone(440.0, seconds=2.0)
    noise = torch.randn(WHISPER_SR)
    config = DegradationConfig()

    first, snr_a = degrade(speech, WHISPER_SR, config, noise, seed=1234)
    second, snr_b = degrade(speech, WHISPER_SR, config, noise, seed=1234)
    other, snr_c = degrade(speech, WHISPER_SR, config, noise, seed=5678)

    assert torch.equal(first, second) and snr_a == snr_b
    assert not torch.equal(first, other), "a different seed must give a different mix"
    assert snr_a != snr_c


def test_degrade_reports_the_snr_it_used() -> None:
    config = DegradationConfig(snr_db_min=8.0, snr_db_max=12.0)
    _, snr = degrade(tone(440.0), WHISPER_SR, config, torch.randn(WHISPER_SR), seed=7)
    assert 8.0 <= snr <= 12.0, "reported SNR must be inside the configured range"


def test_degrade_never_clips() -> None:
    """Output is written as 16-bit PCM; anything above 1.0 would wrap or clamp
    and add distortion that is not part of the modelled channel."""
    loud = tone(440.0) * 0.99
    out, _ = degrade(loud, WHISPER_SR, DegradationConfig(snr_db_min=0.0, snr_db_max=0.0),
                     torch.randn(WHISPER_SR), seed=3)
    assert float(out.abs().max()) <= 1.0


def test_degrade_preserves_length_and_shape() -> None:
    speech = tone(440.0, seconds=2.5)
    out, _ = degrade(speech, WHISPER_SR, DegradationConfig(), torch.randn(WHISPER_SR), seed=11)
    assert out.dim() == 1
    assert abs(out.shape[-1] - speech.shape[-1]) <= 1, "resample round trip may differ by a sample"


def test_stages_can_be_disabled_independently() -> None:
    """The ablation the experiment needs: clean vs noise-only vs full chain."""
    speech = tone(440.0)
    noise = torch.randn(WHISPER_SR)
    off = DegradationConfig(apply_codec=False, apply_noise=False, apply_packet_loss=False)

    out, snr = degrade(speech, WHISPER_SR, off, noise, seed=1)

    assert torch.allclose(out, speech.squeeze(0))
    assert snr == float("inf"), "no noise added means SNR is undefined, not a number"
