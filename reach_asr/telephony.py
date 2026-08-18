"""Turn clean studio speech into something that sounds like an emergency call.

There is no public corpus of real emergency-call audio -- 911 recordings are
legally restricted almost everywhere, for good privacy reasons. So the honest
way to train for that acoustic condition is to *construct* it from a corpus that
is freely available, and to be explicit about what was constructed rather than
recorded.

This module is that construction. Every stage below models a specific, real
property of a phone call placed under duress, and each is applied in the order a
real signal chain would apply it:

1. **Band-limiting to 300-3400 Hz.** The telephone passband, unchanged since
   analogue trunks. This alone destroys most fricative energy -- /s/ and /f/ are
   largely above 4 kHz -- which is why "six" and "fix" are the classic telephone
   confusions and why a model trained on wideband speech degrades on calls.
2. **8 kHz resampling with G.711 mu-law companding.** The actual codec on a
   PSTN call: 8-bit logarithmic quantisation, ~35 dB SNR, coarser in the loud
   passages. Applied *after* band-limiting because that is the physical order --
   quantising first would alias energy the filter should have removed.
3. **Additive environmental noise at a controlled SNR.** Sirens, traffic,
   crying, crowds -- the things actually audible behind an emergency call.
4. **Packet loss.** Short dropouts, which is what a VoIP or cellular call under
   poor signal does. Distinct from noise: the signal is *absent*, not corrupted,
   and models tend to hallucinate fluent text across a gap rather than degrade
   gracefully.

Then back to 16 kHz, because that is what Whisper's feature extractor expects.
Upsampling does not restore the destroyed band -- it just matches the interface.
That is the point: the damage has to be real, or fine-tuning on it teaches
nothing.

Everything is seeded and deterministic given (seed, index), so a training run is
reproducible and the eval set is byte-identical across runs. An augmentation
pipeline that silently varies between the baseline measurement and the
fine-tuned measurement would invalidate the WER comparison it exists to support.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio.functional as AF

# The telephone passband. 300 Hz also removes most handling rumble, which is why
# it was chosen originally -- it is not an arbitrary round number.
PASSBAND_LOW_HZ = 300.0
PASSBAND_HIGH_HZ = 3400.0

# G.711 uses 8-bit mu-law. torchaudio's default quantization_channels is 256,
# stated explicitly here because it is the parameter that makes this G.711
# rather than a generic companding curve.
MU_LAW_CHANNELS = 256

# Biquad sections cascaded to build the passband filter. Each section is
# 2nd-order (12 dB/octave); the counts here were chosen from the measured
# attenuation, not picked for symmetry. See band_limit().
LOWPASS_SECTIONS = 3
HIGHPASS_SECTIONS = 2

TELEPHONY_SR = 8000
WHISPER_SR = 16000


@dataclass(frozen=True)
class DegradationConfig:
    """One reproducible acoustic condition.

    snr_db is sampled per-utterance from [snr_db_min, snr_db_max] rather than
    fixed: a model trained at a single SNR learns that SNR, and the whole point
    is robustness across the range a real call spans. 5-20 dB is the band where
    ASR degrades but a human can still follow the speech -- below ~0 dB the
    reference transcript stops being a fair target.
    """

    snr_db_min: float = 5.0
    snr_db_max: float = 20.0
    packet_loss_rate: float = 0.02
    packet_ms: float = 20.0
    apply_codec: bool = True
    apply_noise: bool = True
    apply_packet_loss: bool = True


def band_limit(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Restrict to the telephone passband.

    Cascaded biquads, not one of each. A single biquad is 2nd-order -- 12
    dB/octave -- which measured only about 19 dB down at 6 kHz against a 3400 Hz
    cutoff, leaving roughly 10% of the amplitude of a tone that a real telephone
    channel removes almost entirely (test_band_limit_keeps_passband_and_kills_the_rest
    caught this). That is not a cosmetic difference: leaving wideband energy in
    means the "telephone" audio still carries the fricative cues that make
    telephone speech hard, so the degradation would be milder than claimed and
    the WER gap it produces would be correspondingly overstated.

    Three cascaded sections give a 6th-order rolloff (~36 dB/octave) on the
    lowpass, which is the order of a real anti-aliasing filter ahead of an 8 kHz
    codec. The highpass gets two sections: the low end matters less because
    there is little speech energy below 300 Hz to begin with.
    """
    out = waveform
    for _ in range(HIGHPASS_SECTIONS):
        out = AF.highpass_biquad(out, sample_rate, cutoff_freq=PASSBAND_LOW_HZ)
    for _ in range(LOWPASS_SECTIONS):
        out = AF.lowpass_biquad(out, sample_rate, cutoff_freq=PASSBAND_HIGH_HZ)
    return out


def apply_g711(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Band-limit, drop to 8 kHz, mu-law quantise, and come back to 16 kHz.

    The round trip is lossy on purpose. Resampling down and back is not a no-op:
    everything above 4 kHz is gone for good, and the mu-law step quantises what
    survives to 8 bits on a logarithmic curve.
    """
    limited = band_limit(waveform, sample_rate)
    narrow = AF.resample(limited, sample_rate, TELEPHONY_SR)

    # mu_law_encoding expects [-1, 1]. Scale down and restore rather than
    # clamping: clamping a hot utterance would add clipping distortion that the
    # codec stage is not supposed to be responsible for.
    peak = narrow.abs().max()
    scale = peak.clamp(min=1e-8)
    encoded = AF.mu_law_encoding(narrow / scale, MU_LAW_CHANNELS)
    decoded = AF.mu_law_decoding(encoded, MU_LAW_CHANNELS) * scale

    return AF.resample(decoded, TELEPHONY_SR, sample_rate)


def mix_noise(
    speech: torch.Tensor, noise: torch.Tensor, snr_db: float, generator: torch.Generator
) -> torch.Tensor:
    """Add `noise` to `speech` at the requested SNR.

    The noise is tiled or randomly cropped to match length. Cropping at a random
    offset matters more than it looks: ESC-50 clips have a characteristic onset,
    and always starting at sample 0 would let the model key on "the siren starts
    exactly when the utterance does" instead of learning noise robustness.
    """
    n_samples = speech.shape[-1]
    noise = noise.reshape(1, -1)

    if noise.shape[-1] < n_samples:
        repeats = (n_samples // noise.shape[-1]) + 1
        noise = noise.repeat(1, repeats)
    if noise.shape[-1] > n_samples:
        max_offset = noise.shape[-1] - n_samples
        offset = int(torch.randint(0, max_offset + 1, (1,), generator=generator).item())
        noise = noise[:, offset : offset + n_samples]

    speech_power = speech.pow(2).mean().clamp(min=1e-10)
    noise_power = noise.pow(2).mean().clamp(min=1e-10)
    # SNR = 10*log10(Ps/Pn)  =>  scale noise so the ratio lands on snr_db.
    target_noise_power = speech_power / (10.0 ** (snr_db / 10.0))
    noise = noise * torch.sqrt(target_noise_power / noise_power)

    return speech + noise


def drop_packets(
    waveform: torch.Tensor,
    sample_rate: int,
    loss_rate: float,
    packet_ms: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """Zero out random fixed-length frames, as a lossy VoIP link would.

    Silence, not noise: a dropped packet means no signal arrived. This is the
    stage most likely to produce Whisper hallucination in the baseline, because
    the decoder's language model happily bridges a gap with fluent invented text
    -- which is precisely the failure worth measuring on emergency audio, where
    an invented street name is far worse than a dropped word.
    """
    if loss_rate <= 0.0:
        return waveform

    packet_len = max(1, int(sample_rate * packet_ms / 1000.0))
    n_packets = waveform.shape[-1] // packet_len
    if n_packets == 0:
        return waveform

    out = waveform.clone()
    keep = torch.rand(n_packets, generator=generator)
    for index in (keep < loss_rate).nonzero(as_tuple=True)[0].tolist():
        start = index * packet_len
        out[..., start : start + packet_len] = 0.0
    return out


def degrade(
    waveform: torch.Tensor,
    sample_rate: int,
    config: DegradationConfig,
    noise: torch.Tensor | None,
    seed: int,
) -> tuple[torch.Tensor, float]:
    """Apply the full chain. Returns (degraded waveform, SNR actually used).

    The SNR is returned rather than only logged so it can be written into the
    manifest -- per-utterance SNR is what makes a WER-vs-SNR breakdown possible
    at eval time, and recovering it afterwards from the audio is not reliable.
    """
    generator = torch.Generator().manual_seed(seed)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    out = waveform

    snr_db = float("inf")
    if config.apply_noise and noise is not None:
        span = config.snr_db_max - config.snr_db_min
        snr_db = config.snr_db_min + float(torch.rand(1, generator=generator).item()) * span
        out = mix_noise(out, noise, snr_db, generator)

    # Codec after noise: on a real call the microphone picks up the room, then
    # the codec encodes whatever it was handed. Encoding clean speech and adding
    # noise afterwards would model a noise source inside the telephone network,
    # which is not the situation being simulated.
    if config.apply_codec:
        out = apply_g711(out, sample_rate)

    if config.apply_packet_loss:
        out = drop_packets(out, sample_rate, config.packet_loss_rate, config.packet_ms, generator)

    # Guard against the sum clipping when stored as 16-bit PCM.
    peak = out.abs().max()
    if peak > 1.0:
        out = out / peak

    return out.squeeze(0), snr_db
