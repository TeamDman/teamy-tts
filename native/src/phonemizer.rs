//! DeepPhonemizer's six-layer forward transformer, expressed as native operations.
use crate::{
    cuda::{Buffer, Device},
    weights::Weights,
};
use anyhow::{Context, Result, ensure};
use std::rc::Rc;

const TEXT_SYMBOLS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüÄÖÜß'";
const PHONEME_SYMBOLS: &str = "abdefghijklmnoprstuvwxyzæçðøŋœɐɑɔəɛɝɹɡɪʁʃʊʌʏʒʔː͡θ";

struct Linear {
    weight: Buffer,
    bias: Buffer,
    input: usize,
    output: usize,
}
impl Linear {
    fn load(w: &Weights, d: &Rc<Device>, name: &str, input: usize, output: usize) -> Result<Self> {
        Ok(Self {
            weight: d.upload(w.f32(&format!("phonemizer.{name}.weight"), &[output, input])?)?,
            bias: d.upload(w.f32(&format!("phonemizer.{name}.bias"), &[output])?)?,
            input,
            output,
        })
    }
    fn run(&self, x: &Buffer, time: usize, relu_input: bool) -> Result<Buffer> {
        x.conv(
            &self.weight,
            &self.bias,
            None,
            self.input,
            self.output,
            time,
            1,
            1,
            if relu_input { 0. } else { 1. },
        )
    }
}
struct Norm {
    gamma: Buffer,
    beta: Buffer,
}
impl Norm {
    fn load(w: &Weights, d: &Rc<Device>, name: &str) -> Result<Self> {
        Ok(Self {
            gamma: d.upload(w.f32(&format!("phonemizer.{name}.gamma"), &[512])?)?,
            beta: d.upload(w.f32(&format!("phonemizer.{name}.beta"), &[512])?)?,
        })
    }
    fn run(&self, x: &Buffer, residual: Option<&Buffer>, time: usize) -> Result<Buffer> {
        x.layer_norm(residual, &self.gamma, &self.beta, time)
    }
}
struct Encoder {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    linear1: Linear,
    linear2: Linear,
    norm1: Norm,
    norm2: Norm,
}
impl Encoder {
    fn load(w: &Weights, d: &Rc<Device>, i: usize) -> Result<Self> {
        let p = format!("encoder.{i}");
        Ok(Self {
            query: Linear::load(w, d, &format!("{p}.self_attn.query"), 512, 512)?,
            key: Linear::load(w, d, &format!("{p}.self_attn.key"), 512, 512)?,
            value: Linear::load(w, d, &format!("{p}.self_attn.value"), 512, 512)?,
            out: Linear::load(w, d, &format!("{p}.self_attn.out_proj"), 512, 512)?,
            linear1: Linear::load(w, d, &format!("{p}.linear1"), 512, 1024)?,
            linear2: Linear::load(w, d, &format!("{p}.linear2"), 1024, 512)?,
            norm1: Norm::load(w, d, &format!("{p}.norm1"))?,
            norm2: Norm::load(w, d, &format!("{p}.norm2"))?,
        })
    }
    fn run(&self, x: &Buffer, time: usize) -> Result<Buffer> {
        let q = self.query.run(x, time, false)?;
        let k = self.key.run(x, time, false)?;
        let v = self.value.run(x, time, false)?;
        let attention = self.out.run(&q.attention(&k, &v, time)?, time, false)?;
        let x = self.norm1.run(x, Some(&attention), time)?;
        let ff = self
            .linear2
            .run(&self.linear1.run(&x, time, false)?, time, true)?;
        self.norm2.run(&x, Some(&ff), time)
    }
}
pub struct Phonemizer {
    device: Rc<Device>,
    embedding: Buffer,
    position: Buffer,
    scale: f32,
    encoder: Vec<Encoder>,
    norm: Norm,
    out: Linear,
}
#[cfg(test)]
mod tests {
    use super::Phonemizer;
    #[test]
    fn text_symbols_and_ctc_decoding_preserve_product_contract() {
        assert_eq!(Phonemizer::tokens("a").unwrap(), [2, 4, 4, 4, 3]);
        assert_eq!(Phonemizer::decode(&[2, 4, 0, 4, 5, 3, 6]).unwrap(), "ab");
        assert!(Phonemizer::decode(&[0, 1, 2, 3]).is_err());
        assert!(Phonemizer::tokens("").is_err());
        assert!(Phonemizer::tokens("💥").is_err());
        assert!(Phonemizer::tokens(&"a".repeat(1667)).is_err());
    }
}
impl Phonemizer {
    pub fn load(w: &Weights, d: Rc<Device>) -> Result<Self> {
        let scale = w.values("phonemizer.pos_encoder.scale")?;
        ensure!(scale.len() == 1, "position scale shape");
        Ok(Self {
            embedding: d.upload(w.f32("phonemizer.embedding.weight", &[64, 512])?)?,
            position: d.upload(w.f32("phonemizer.pos_encoder.pe", &[5000, 1, 512])?)?,
            scale: scale[0],
            encoder: (0..6)
                .map(|i| Encoder::load(w, &d, i))
                .collect::<Result<_>>()?,
            norm: Norm::load(w, &d, "norm")?,
            out: Linear::load(w, &d, "fc_out", 512, 53)?,
            device: d,
        })
    }
    pub fn tokens(word: &str) -> Result<Vec<i32>> {
        ensure!(
            !word.is_empty() && word.chars().count() <= 1666,
            "phonemizer word length outside 1..1666"
        );
        let mut ids = vec![2];
        for c in word.chars() {
            let id = TEXT_SYMBOLS
                .chars()
                .position(|s| s == c)
                .with_context(|| format!("unsupported phonemizer character {c:?}"))?
                as i32
                + 4;
            ids.extend([id; 3]);
        }
        ids.push(3);
        Ok(ids)
    }
    pub fn forward(&self, ids: &[i32]) -> Result<Buffer> {
        ensure!(
            !ids.is_empty() && ids.len() <= 5000 && ids.iter().all(|&id| id > 0 && id < 64),
            "invalid unpadded phonemizer sequence"
        );
        let time = ids.len();
        let bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
        let ids = self.device.upload(&bytes)?;
        let mut x = self
            .embedding
            .phoneme_embedding(&ids, &self.position, self.scale, time)?;
        for layer in &self.encoder {
            x = layer.run(&x, time)?;
        }
        self.out.run(&self.norm.run(&x, None, time)?, time, false)
    }
    pub fn decode(ids: &[i32]) -> Result<String> {
        let mut output = String::new();
        let mut previous = None;
        for &id in ids {
            if id == 3 {
                break;
            }
            if id <= 2 || previous == Some(id) {
                continue;
            }
            previous = Some(id);
            let c = PHONEME_SYMBOLS
                .chars()
                .nth((id - 4) as usize)
                .context("phoneme outside vocabulary")?;
            output.push(c);
        }
        ensure!(!output.is_empty(), "phonemizer produced no phonemes");
        Ok(output)
    }
    pub fn phonemize_word(&self, word: &str) -> Result<String> {
        let ids = Self::tokens(word)?;
        Self::decode(&self.forward(&ids)?.argmax(ids.len(), 53)?.download_i32()?)
    }
}
