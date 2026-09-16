// Shared PCM encoding for both inference builds.
pub(crate) fn encode_pcm16_wav(sample_rate_hz: u32, samples: &[f32]) -> eyre::Result<Vec<u8>> {
    let data_bytes = samples
        .len()
        .checked_mul(2)
        .ok_or_else(|| eyre::eyre!("WAV data size overflows usize"))?;
    let riff_size = 36usize
        .checked_add(data_bytes)
        .ok_or_else(|| eyre::eyre!("WAV RIFF size overflows usize"))?;
    let riff_size =
        u32::try_from(riff_size).map_err(|error| eyre::eyre!("WAV is too large: {error}"))?;
    let data_bytes =
        u32::try_from(data_bytes).map_err(|error| eyre::eyre!("WAV is too large: {error}"))?;
    let byte_rate = sample_rate_hz
        .checked_mul(2)
        .ok_or_else(|| eyre::eyre!("WAV byte rate overflows u32"))?;

    let mut bytes = Vec::with_capacity(44 + samples.len() * 2);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&riff_size.to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&sample_rate_hz.to_le_bytes());
    bytes.extend_from_slice(&byte_rate.to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&data_bytes.to_le_bytes());
    for &sample in samples {
        let sample = if sample.is_finite() {
            sample.clamp(-1.0, 1.0)
        } else {
            0.0
        };
        #[expect(
            clippy::cast_possible_truncation,
            reason = "The sample was clamped to the signed 16-bit audio range before conversion."
        )]
        let integer = (sample * 32_767.0).round() as i16;
        bytes.extend_from_slice(&integer.to_le_bytes());
    }

    Ok(bytes)
}
