//! The deployed HiFiGAN architecture as ordinary Rust code.
use crate::{
    cuda::{Buffer, Device},
    weights::Weights,
};
use anyhow::{Result, ensure};
use std::rc::Rc;

#[derive(Debug)]
struct Conv {
    w: Buffer,
    b: Buffer,
    input: usize,
    output: usize,
    kernel: usize,
    dilation: usize,
}
impl Conv {
    fn load(
        weights: &Weights,
        device: &Rc<Device>,
        name: &str,
        input: usize,
        output: usize,
        kernel: usize,
        dilation: usize,
    ) -> Result<Self> {
        Ok(Self {
            w: device.upload(
                weights.f32(&format!("vocoder.{name}.weight"), &[output, input, kernel])?,
            )?,
            b: device.upload(weights.f32(&format!("vocoder.{name}.bias"), &[output])?)?,
            input,
            output,
            kernel,
            dilation,
        })
    }
    fn run(
        &self,
        x: &Buffer,
        time: usize,
        slope: f32,
        residual: Option<&Buffer>,
    ) -> Result<Buffer> {
        x.conv(
            &self.w,
            &self.b,
            residual,
            self.input,
            self.output,
            time,
            self.kernel,
            self.dilation,
            slope,
        )
    }
}
#[derive(Debug)]
struct Residual {
    layers: Vec<(Conv, Conv)>,
}
impl Residual {
    fn run(&self, x: &Buffer, time: usize) -> Result<Buffer> {
        let mut output = None;
        for (first, second) in &self.layers {
            let input = output.as_ref().unwrap_or(x);
            let hidden = first.run(input, time, 0.1, None)?;
            output = Some(second.run(&hidden, time, 0.1, Some(input))?);
        }
        Ok(output.expect("three residual layers"))
    }
}
#[derive(Debug)]
struct UpStage {
    w: Buffer,
    b: Buffer,
    input: usize,
    output: usize,
    stride: usize,
    blocks: Vec<Residual>,
}

#[derive(Debug)]
pub struct HiFiGan {
    device: Rc<Device>,
    pre: Conv,
    stages: Vec<UpStage>,
    post: Conv,
}
impl HiFiGan {
    pub fn load(weights: &Weights, device: Rc<Device>) -> Result<Self> {
        let pre = Conv::load(weights, &device, "conv_pre", 80, 512, 7, 1)?;
        let mut stages = Vec::new();
        for (stage, (input, output, stride)) in
            [(512, 256, 8), (256, 128, 8), (128, 64, 2), (64, 32, 2)]
                .into_iter()
                .enumerate()
        {
            let name = format!("vocoder.ups.{stage}.weight");
            weights.f32(&name, &[input, output, 2 * stride])?;
            let source = weights.values(&name)?;
            // Pack [in,out,k] into [phase,out,in,tap]; two taps per phase.
            let mut packed = vec![0.; stride * output * input * 2];
            for phase in 0..stride {
                for o in 0..output {
                    for i in 0..input {
                        for tap in 0..2 {
                            let k = (phase + stride / 2) % stride + tap * stride;
                            packed[((phase * output + o) * input + i) * 2 + tap] =
                                source[(i * output + o) * 2 * stride + k];
                        }
                    }
                }
            }
            let w = device.upload_f32(&packed)?;
            let b = device.upload(weights.f32(&format!("vocoder.ups.{stage}.bias"), &[output])?)?;
            let mut blocks = Vec::new();
            for (block, kernel) in [3, 7, 11].into_iter().enumerate() {
                let mut layers = Vec::new();
                for (layer, dilation) in [1, 3, 5].into_iter().enumerate() {
                    let first = Conv::load(
                        weights,
                        &device,
                        &format!("resblocks.{}.convs1.{layer}", stage * 3 + block),
                        output,
                        output,
                        kernel,
                        dilation,
                    )?;
                    let second = Conv::load(
                        weights,
                        &device,
                        &format!("resblocks.{}.convs2.{layer}", stage * 3 + block),
                        output,
                        output,
                        kernel,
                        1,
                    )?;
                    layers.push((first, second));
                }
                blocks.push(Residual { layers });
            }
            stages.push(UpStage {
                w,
                b,
                input,
                output,
                stride,
                blocks,
            });
        }
        let post = Conv::load(weights, &device, "conv_post", 32, 1, 7, 1)?;
        Ok(Self {
            device,
            pre,
            stages,
            post,
        })
    }

    pub fn synthesize(&self, mel: &[f32], frames: usize) -> Result<Vec<f32>> {
        ensure!(
            frames > 0 && mel.len() == 80 * frames,
            "expected nonempty [80,frames] mel"
        );
        let input = self.device.upload_f32(mel)?;
        self.synthesize_device(&input, frames)?.download()
    }
    pub fn synthesize_device(&self, mel: &Buffer, mut frames: usize) -> Result<Buffer> {
        ensure!(
            frames > 0 && mel.len == 80 * frames,
            "device mel dimensions"
        );
        let mut x = self.pre.run(mel, frames, 1.0, None)?;
        for stage in &self.stages {
            x = x.up(
                &stage.w,
                &stage.b,
                stage.input,
                stage.output,
                frames,
                stage.stride,
            )?;
            frames *= stage.stride;
            let a = stage.blocks[0].run(&x, frames)?;
            let b = stage.blocks[1].run(&x, frames)?;
            let c = stage.blocks[2].run(&x, frames)?;
            x = a.mean3(&b, &c)?;
        }
        // The deployed final activation uses PyTorch's default slope 0.01.
        self.post.run(&x, frames, 0.01, None)?.tanh()
    }
}
