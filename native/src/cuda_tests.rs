//! Small independent double-precision CPU oracles for GPU boundary cases.
use crate::cuda::Device;

fn values(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 17 % 43) as f32 - 21.) / 37.).collect()
}
fn close(actual: &[f32], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && (f64::from(a) - e).abs() <= tolerance,
            "element {i}: {a} != {e}"
        );
    }
}
#[test]
#[ignore = "requires a CUDA GPU and runtime DLLs"]
fn convolution_even_padding_dilation_and_short_sequences() {
    let d = Device::new().unwrap();
    for time in [1, 2, 3, 13, 33] {
        for kernel in [1, 2, 3, 8] {
            for dilation in [1, 3, 5] {
                let (input, output) = (3, 5);
                let x = values(input * time);
                let w = values(output * input * kernel);
                let b = values(output);
                let r = values(output * time);
                let actual = d
                    .upload_f32(&x)
                    .unwrap()
                    .conv(
                        &d.upload_f32(&w).unwrap(),
                        &d.upload_f32(&b).unwrap(),
                        Some(&d.upload_f32(&r).unwrap()),
                        input,
                        output,
                        time,
                        kernel,
                        dilation,
                        0.1,
                    )
                    .unwrap()
                    .download()
                    .unwrap();
                let mut expected = vec![0.; output * time];
                for o in 0..output {
                    for t in 0..time {
                        let mut sum = f64::from(b[o]) + f64::from(r[o * time + t]);
                        for c in 0..input {
                            for k in 0..kernel {
                                let src = t as isize + (k * dilation) as isize
                                    - ((kernel / 2) * dilation) as isize;
                                if src >= 0 && src < time as isize {
                                    let v = f64::from(x[c * time + src as usize]);
                                    sum +=
                                        v.max(v * 0.1) * f64::from(w[(o * input + c) * kernel + k]);
                                }
                            }
                        }
                        expected[o * time + t] = sum;
                    }
                }
                close(&actual, &expected, 2e-5);
            }
        }
    }
}
#[test]
#[ignore = "requires a CUDA GPU and runtime DLLs"]
fn transpose_convolution_polyphase_matches_direct_scatter() {
    let d = Device::new().unwrap();
    for time in [1, 2, 7, 33] {
        for stride in [2, 8] {
            let (input, output) = (3, 5);
            let kernel = 2 * stride;
            let n = time * stride;
            let x = values(input * time);
            let w = values(input * output * kernel);
            let b = values(output);
            let mut packed = vec![0.; w.len()];
            for p in 0..stride {
                for o in 0..output {
                    for c in 0..input {
                        for tap in 0..2 {
                            let k = (p + stride / 2) % stride + tap * stride;
                            packed[((p * output + o) * input + c) * 2 + tap] =
                                w[(c * output + o) * kernel + k];
                        }
                    }
                }
            }
            let actual = d
                .upload_f32(&x)
                .unwrap()
                .up(
                    &d.upload_f32(&packed).unwrap(),
                    &d.upload_f32(&b).unwrap(),
                    input,
                    output,
                    time,
                    stride,
                )
                .unwrap()
                .download()
                .unwrap();
            let mut expected = (0..output * n)
                .map(|i| f64::from(b[i / n]))
                .collect::<Vec<_>>();
            for c in 0..input {
                for t in 0..time {
                    for o in 0..output {
                        for k in 0..kernel {
                            let dst = (t * stride + k) as isize - (stride / 2) as isize;
                            if dst >= 0 && dst < n as isize {
                                let v = f64::from(x[c * time + t]);
                                expected[o * n + dst as usize] +=
                                    v.max(v * 0.1) * f64::from(w[(c * output + o) * kernel + k]);
                            }
                        }
                    }
                }
            }
            close(&actual, &expected, 2e-5);
        }
    }
}
#[test]
#[ignore = "requires a CUDA GPU and runtime DLLs"]
fn attention_and_layernorm_match_double_precision_reference() {
    let d = Device::new().unwrap();
    for time in [1, 5, 33] {
        let q = values(512 * time);
        let k = q.iter().rev().copied().collect::<Vec<_>>();
        let v = q.iter().map(|x| x * 0.7).collect::<Vec<_>>();
        let actual = d
            .upload_f32(&q)
            .unwrap()
            .attention(&d.upload_f32(&k).unwrap(), &d.upload_f32(&v).unwrap(), time)
            .unwrap()
            .download()
            .unwrap();
        let mut expected = vec![0.; 512 * time];
        for head in 0..4 {
            for query in 0..time {
                let mut scores = (0..time)
                    .map(|key| {
                        (0..128)
                            .map(|c| {
                                f64::from(q[(head * 128 + c) * time + query])
                                    * f64::from(k[(head * 128 + c) * time + key])
                            })
                            .sum::<f64>()
                            / 11.313708
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                for s in &mut scores {
                    *s = (*s - max).exp();
                }
                let sum = scores.iter().sum::<f64>();
                for c in 0..128 {
                    expected[(head * 128 + c) * time + query] = (0..time)
                        .map(|key| scores[key] / sum * f64::from(v[(head * 128 + c) * time + key]))
                        .sum();
                }
            }
        }
        close(&actual, &expected, 2e-5);
        let gamma = values(512);
        let beta = gamma.iter().map(|v| v * 0.2).collect::<Vec<_>>();
        let actual = d
            .upload_f32(&q)
            .unwrap()
            .layer_norm(
                Some(&d.upload_f32(&v).unwrap()),
                &d.upload_f32(&gamma).unwrap(),
                &d.upload_f32(&beta).unwrap(),
                time,
            )
            .unwrap()
            .download()
            .unwrap();
        for t in 0..time {
            let vals = (0..512)
                .map(|c| f64::from(q[c * time + t] + v[c * time + t]))
                .collect::<Vec<_>>();
            let mean = vals.iter().sum::<f64>() / 512.;
            let variance = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / 512.;
            for c in 0..512 {
                expected[c * time + t] = (vals[c] - mean) / (variance + 1e-5).sqrt()
                    * f64::from(gamma[c])
                    + f64::from(beta[c]);
            }
        }
        close(&actual, &expected, 2e-5);
    }
}
#[test]
#[ignore = "requires a CUDA GPU and runtime DLLs"]
fn transpose_pool_and_gather_preserve_boundary_indices() {
    let d = Device::new().unwrap();
    for (channels, time) in [(1, 1), (3, 2), (31, 33), (33, 31)] {
        let x = values(channels * time);
        let gpu = d.upload_f32(&x).unwrap();
        let expected = (0..time)
            .flat_map(|t| {
                (0..channels).map({
                    let x = &x;
                    move |c| f64::from(x[c * time + t])
                })
            })
            .collect::<Vec<_>>();
        close(
            &gpu.transpose(channels, time).unwrap().download().unwrap(),
            &expected,
            0.,
        );
        let expected = (0..x.len())
            .map(|i| {
                f64::from(if i % time == 0 {
                    x[i]
                } else {
                    x[i].max(x[i - 1])
                })
            })
            .collect::<Vec<_>>();
        close(
            &gpu.max_pool(time).unwrap().download().unwrap(),
            &expected,
            0.,
        );
        let indices = [(time - 1) as i32, 0, 0];
        let expected = (0..channels)
            .flat_map(|c| {
                indices.iter().map({
                    let x = &x;
                    move |&t| f64::from(x[c * time + t as usize])
                })
            })
            .collect::<Vec<_>>();
        close(
            &gpu.gather(&indices, time, channels)
                .unwrap()
                .download()
                .unwrap(),
            &expected,
            0.,
        );
        assert!(gpu.gather(&[-1], time, channels).is_err());
        assert!(gpu.gather(&[time as i32], time, channels).is_err());
    }
}
