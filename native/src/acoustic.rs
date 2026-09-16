//! Source-defined deployed MultiForwardTacotron, batch one, inference only.
//! All feature planes use [channel,time]; cuDNN boundaries transpose explicitly.
use crate::{
    cuda::{Buffer, Device},
    rnn::{Dnn, Rnn},
    weights::Weights,
};
use anyhow::{Context, Result, ensure};
use std::rc::Rc;

struct Conv {
    w: Buffer,
    b: Buffer,
    input: usize,
    output: usize,
    kernel: usize,
}
impl Conv {
    fn load(weights: &Weights, device: &Rc<Device>, prefix: &str, linear: bool) -> Result<Self> {
        let key = format!("acoustic.{prefix}.weight");
        let shape = weights.shape(&key)?;
        ensure!(
            shape.len() == if linear { 2 } else { 3 },
            "{prefix}: wrong rank"
        );
        let (output, input, kernel) = (shape[0], shape[1], if linear { 1 } else { shape[2] });
        let w = device.upload(weights.f32(&key, shape)?)?;
        let bias = format!("acoustic.{prefix}.bias");
        let b = if weights.contains(&bias) {
            device.upload(weights.f32(&bias, &[output])?)?
        } else {
            device.upload_f32(&vec![0.; output])?
        };
        Ok(Self {
            w,
            b,
            input,
            output,
            kernel,
        })
    }
    fn run(&self, x: &Buffer, time: usize) -> Result<Buffer> {
        x.conv(
            &self.w,
            &self.b,
            None,
            self.input,
            self.output,
            time,
            self.kernel,
            1,
            1.0,
        )
    }
}
struct NormConv {
    conv: Conv,
    scale: Buffer,
    bias: Buffer,
    relu: bool,
}
impl NormConv {
    fn load(weights: &Weights, device: &Rc<Device>, prefix: &str, relu: bool) -> Result<Self> {
        let conv = Conv::load(weights, device, &format!("{prefix}.conv"), false)?;
        let get = |name: &str| weights.values(&format!("acoustic.{prefix}.bnorm.{name}"));
        let gamma = get("weight")?;
        let beta = get("bias")?;
        let mean = get("running_mean")?;
        let variance = get("running_var")?;
        ensure!(
            [gamma.len(), beta.len(), mean.len(), variance.len()]
                .iter()
                .all(|&n| n == conv.output),
            "batch norm shape"
        );
        let scale: Vec<f32> = gamma
            .iter()
            .zip(&variance)
            .map(|(g, v)| g / (v + 1e-5).sqrt())
            .collect();
        let bias: Vec<f32> = beta
            .iter()
            .zip(&mean)
            .zip(&scale)
            .map(|((b, m), s)| b - m * s)
            .collect();
        Ok(Self {
            conv,
            scale: device.upload_f32(&scale)?,
            bias: device.upload_f32(&bias)?,
            relu,
        })
    }
    fn run(&self, x: &Buffer, time: usize) -> Result<Buffer> {
        // ReLU is BEFORE normalization in this model. Do not fold BN through it.
        self.conv
            .run(x, time)?
            .batch_norm(&self.scale, &self.bias, time, self.relu)
    }
}
struct Embedding {
    w: Buffer,
    width: usize,
    vocab: usize,
}
impl Embedding {
    fn load(weights: &Weights, device: &Rc<Device>, prefix: &str) -> Result<Self> {
        let key = format!("acoustic.{prefix}.weight");
        let shape = weights.shape(&key)?;
        ensure!(shape.len() == 2, "embedding rank");
        Ok(Self {
            w: device.upload(weights.f32(&key, shape)?)?,
            vocab: shape[0],
            width: shape[1],
        })
    }
    fn run(&self, ids: &Buffer, time: usize) -> Result<Buffer> {
        self.w.embedding(ids, time, self.width)
    }
}
fn recurrent(rnn: &Rnn, x: &Buffer, time: usize) -> Result<Buffer> {
    let input = x.transpose(rnn.input, time)?;
    rnn.run(&input, time)?.transpose(time, 2 * rnn.hidden)
}
struct Predictor {
    embedding: Embedding,
    condition: Option<Embedding>,
    convs: Vec<NormConv>,
    rnn: Rnn,
    linear: Conv,
}
impl Predictor {
    fn load(weights: &Weights, api: &Rc<Dnn>, prefix: &str, conditional: bool) -> Result<Self> {
        let device = &api.device;
        let embedding = Embedding::load(weights, device, &format!("{prefix}.embedding"))?;
        ensure!(embedding.vocab == 135, "predictor vocabulary");
        let condition = if conditional {
            let e = Embedding::load(weights, device, &format!("{prefix}.pitch_cond_embedding"))?;
            ensure!(e.vocab == 4, "pitch condition embedding vocabulary");
            Some(e)
        } else {
            None
        };
        let convs = (0..3)
            .map(|i| NormConv::load(weights, device, &format!("{prefix}.convs.{i}"), true))
            .collect::<Result<_>>()?;
        let rnn = Rnn::load(
            weights,
            api.clone(),
            &format!("acoustic.{prefix}.rnn"),
            false,
        )?;
        let linear = Conv::load(weights, device, &format!("{prefix}.lin"), true)?;
        Ok(Self {
            embedding,
            condition,
            convs,
            rnn,
            linear,
        })
    }
    fn run(
        &self,
        tokens: &Buffer,
        condition: Option<&Buffer>,
        speaker: &Buffer,
        time: usize,
    ) -> Result<Buffer> {
        let embedding = self.embedding.run(tokens, time)?;
        let condition = match &self.condition {
            Some(e) => Some(e.run(condition.context("missing pitch condition")?, time)?),
            None => None,
        };
        let mut parts = vec![&embedding];
        if let Some(ref c) = condition {
            parts.push(c);
        }
        parts.push(speaker);
        let mut x = Buffer::concat(&parts)?;
        for conv in &self.convs {
            x = conv.run(&x, time)?;
        }
        self.linear.run(&recurrent(&self.rnn, &x, time)?, time)
    }
}
struct Highway {
    a: Conv,
    b: Conv,
}
struct Cbhg {
    bank: Vec<NormConv>,
    project1: NormConv,
    project2: NormConv,
    pre: Conv,
    highways: Vec<Highway>,
    rnn: Rnn,
}
impl Cbhg {
    fn load(weights: &Weights, api: &Rc<Dnn>, prefix: &str, bank_size: usize) -> Result<Self> {
        let device = &api.device;
        let bank = (0..bank_size)
            .map(|i| NormConv::load(weights, device, &format!("{prefix}.conv1d_bank.{i}"), true))
            .collect::<Result<_>>()?;
        let project1 = NormConv::load(weights, device, &format!("{prefix}.conv_project1"), true)?;
        let project2 = NormConv::load(weights, device, &format!("{prefix}.conv_project2"), false)?;
        let pre = Conv::load(weights, device, &format!("{prefix}.pre_highway"), true)?;
        let highways = (0..4)
            .map(|i| {
                Ok(Highway {
                    a: Conv::load(weights, device, &format!("{prefix}.highways.{i}.W1"), true)?,
                    b: Conv::load(weights, device, &format!("{prefix}.highways.{i}.W2"), true)?,
                })
            })
            .collect::<Result<_>>()?;
        let rnn = Rnn::load(
            weights,
            api.clone(),
            &format!("acoustic.{prefix}.rnn"),
            false,
        )?;
        Ok(Self {
            bank,
            project1,
            project2,
            pre,
            highways,
            rnn,
        })
    }
    fn run(&self, input: &Buffer, time: usize) -> Result<Buffer> {
        let bank = self
            .bank
            .iter()
            .map(|conv| conv.run(input, time))
            .collect::<Result<Vec<_>>>()?;
        let x = Buffer::concat(&bank.iter().collect::<Vec<_>>())?.max_pool(time)?;
        let x = self
            .project2
            .run(&self.project1.run(&x, time)?, time)?
            .add(input)?;
        let mut x = self.pre.run(&x, time)?;
        for h in &self.highways {
            x = x.highway(&h.a.run(&x, time)?, &h.b.run(&x, time)?)?;
        }
        recurrent(&self.rnn, &x, time)
    }
}

pub struct AcousticOutput {
    pub mel: Buffer,
    pub raw_mel: Buffer,
    pub frames: usize,
    pub durations: Vec<f32>,
}
pub struct ForwardTacotron {
    device: Rc<Device>,
    embedding: Embedding,
    pitch_condition: Predictor,
    duration: Predictor,
    pitch: Predictor,
    energy: Predictor,
    prenet: Cbhg,
    lstm: Rnn,
    linear: Conv,
    postnet: Cbhg,
    post_projection: Conv,
    pitch_projection: Conv,
    energy_projection: Conv,
    p1: Buffer,
    p2: Buffer,
}
impl ForwardTacotron {
    pub fn load(weights: &Weights, api: Rc<Dnn>) -> Result<Self> {
        let device = api.device.clone();
        let embedding = Embedding::load(weights, &device, "embedding")?;
        ensure!(
            embedding.vocab == 135 && embedding.width == 256,
            "GLaDOS embedding shape"
        );
        Ok(Self {
            embedding,
            pitch_condition: Predictor::load(weights, &api, "pitch_cond_pred", false)?,
            duration: Predictor::load(weights, &api, "dur_pred", true)?,
            pitch: Predictor::load(weights, &api, "pitch_pred", true)?,
            energy: Predictor::load(weights, &api, "energy_pred", false)?,
            prenet: Cbhg::load(weights, &api, "prenet", 16)?,
            lstm: Rnn::load(weights, api.clone(), "acoustic.lstm", true)?,
            linear: Conv::load(weights, &device, "lin", true)?,
            postnet: Cbhg::load(weights, &api, "postnet", 8)?,
            post_projection: Conv::load(weights, &device, "post_proj", true)?,
            pitch_projection: Conv::load(weights, &device, "pitch_proj", false)?,
            energy_projection: Conv::load(weights, &device, "energy_proj", false)?,
            p1: device.upload(weights.f32("voice.p1", &[1, 256])?)?,
            p2: device.upload(weights.f32("voice.p2", &[1, 256])?)?,
            device,
        })
    }
    pub fn generate(&self, tokens: &[i32], voice: &str, alpha: f32) -> Result<AcousticOutput> {
        ensure!(
            tokens.len() >= 2 && tokens.len() <= 65535,
            "token count outside supported range"
        );
        ensure!(
            tokens
                .iter()
                .all(|&t| t >= 0 && (t as usize) < self.embedding.vocab),
            "token outside vocabulary"
        );
        ensure!(
            alpha.is_finite() && alpha > 0.,
            "alpha must be finite and positive"
        );
        let time = tokens.len();
        let token_bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        let token_ids = self.device.upload(&token_bytes)?;
        let voice = match voice {
            "p1" => &self.p1,
            "p2" => &self.p2,
            _ => anyhow::bail!("unknown voice"),
        };
        let speaker = voice.broadcast(time)?;
        let condition = self
            .pitch_condition
            .run(&token_ids, None, &speaker, time)?
            .argmax(time, 3)?;
        let durations = self
            .duration
            .run(&token_ids, Some(&condition), &speaker, time)?;
        let pitch = self
            .pitch
            .run(&token_ids, Some(&condition), &speaker, time)?;
        let energy = self.energy.run(&token_ids, None, &speaker, time)?;
        let encoded = self
            .prenet
            .run(&self.embedding.run(&token_ids, time)?, time)?;
        let conditioned = Buffer::concat(&[&encoded, &speaker])?
            .add(&self.pitch_projection.run(&pitch, time)?)?
            .add(&self.energy_projection.run(&energy, time)?)?;
        // Both pitch_strength and energy_strength in the deployed model are 1.
        let mut durations = durations.download()?;
        for duration in &mut durations {
            *duration /= alpha;
        }
        ensure!(
            durations.iter().all(|d| d.is_finite()),
            "nonfinite duration prediction"
        );
        // generate_jit tests the sum after truncation, before clamping negatives.
        if durations.iter().map(|&d| d as i64).sum::<i64>() <= 0 {
            durations.fill(2.);
        }
        let mut indices = Vec::new();
        for (i, duration) in durations.iter_mut().enumerate() {
            *duration = duration.max(0.);
            let count = (*duration + 0.5) as usize;
            ensure!(
                count <= 65535 && indices.len() + count <= 65535,
                "duration expansion exceeds cuDNN sequence limit"
            );
            indices.extend(std::iter::repeat_n(i as i32, count));
        }
        let frames = indices.len();
        ensure!(frames > 0, "zero acoustic frames");
        let expanded = conditioned.gather(&indices, time, 768)?;
        let decoded = recurrent(&self.lstm, &expanded, frames)?;
        let raw_mel = self.linear.run(&decoded, frames)?;
        let post = self.postnet.run(&raw_mel, frames)?;
        let mel = self.post_projection.run(&post, frames)?;
        Ok(AcousticOutput {
            mel,
            raw_mel,
            frames,
            durations,
        })
    }
}
