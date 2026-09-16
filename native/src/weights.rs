use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use std::{collections::HashMap, fs::File, path::Path};

#[derive(Debug)]
struct Entry {
    shape: Vec<usize>,
    dtype: Dtype,
    start: usize,
    len: usize,
}

/// Immutable tensor artifact. The validated index is built once at load.
#[derive(Debug)]
pub struct Weights {
    map: Mmap,
    entries: HashMap<String, Entry>,
}

impl Weights {
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }
    pub fn integers(&self, name: &str) -> Result<Vec<i32>> {
        let entry = self.entries.get(name).context("missing integer tensor")?;
        ensure!(entry.dtype == Dtype::I64, "expected I64");
        self.map[entry.start..entry.start + entry.len]
            .chunks_exact(8)
            .map(|b| Ok(i64::from_le_bytes(b.try_into().unwrap()).try_into()?))
            .collect()
    }
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        // SAFETY: model artifacts are immutable for the lifetime of the runtime.
        let map = unsafe { Mmap::map(&file)? };
        let (_, metadata) = SafeTensors::read_metadata(&map).context("validate tensor metadata")?;
        ensure!(
            metadata
                .metadata()
                .as_ref()
                .and_then(|m| m.get("format"))
                .is_some_and(|v| v == "teamy-glados-native-v1"),
            "unsupported native tensor artifact format"
        );
        let archive = SafeTensors::deserialize(&map).context("validate tensor archive")?;
        let entries = archive
            .tensors()
            .into_iter()
            .map(|(name, view)| {
                let start = view.data().as_ptr() as usize - map.as_ptr() as usize;
                (
                    name,
                    Entry {
                        shape: view.shape().to_vec(),
                        dtype: view.dtype(),
                        start,
                        len: view.data().len(),
                    },
                )
            })
            .collect();
        Ok(Self { map, entries })
    }

    pub fn f32(&self, name: &str, shape: &[usize]) -> Result<&[u8]> {
        let entry = self
            .entries
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?;
        ensure!(
            entry.dtype == Dtype::F32,
            "{name}: expected F32, got {:?}",
            entry.dtype
        );
        ensure!(
            entry.shape == shape,
            "{name}: expected {shape:?}, got {:?}",
            entry.shape
        );
        Ok(&self.map[entry.start..entry.start + entry.len])
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(&self
            .entries
            .get(name)
            .with_context(|| format!("missing tensor {name}"))?
            .shape)
    }

    pub fn values(&self, name: &str) -> Result<Vec<f32>> {
        let shape = self.shape(name)?;
        self.f32(name, shape)?
            .chunks_exact(4)
            .map(|bytes| {
                let value = f32::from_le_bytes(bytes.try_into().unwrap());
                if !value.is_finite() {
                    bail!("{name}: nonfinite tensor value");
                }
                Ok(value)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::Weights;
    use std::{
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };
    struct Fixture(PathBuf);
    impl Fixture {
        fn new(format: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "glados-weight-test-{}-{}.safetensors",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
            let mut header=serde_json::to_vec(&serde_json::json!({"__metadata__":{"format":format},"value":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})).unwrap();
            header.resize(header.len().next_multiple_of(8), b' ');
            let mut data = (header.len() as u64).to_le_bytes().to_vec();
            data.extend(header);
            data.extend(1.25f32.to_le_bytes());
            data.extend((-2.5f32).to_le_bytes());
            std::fs::write(&path, data).unwrap();
            Self(path)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_file(&self.0).unwrap();
        }
    }
    #[test]
    fn archive_shape_dtype_and_format_are_checked() {
        let fixture = Fixture::new("teamy-glados-native-v1");
        let w = Weights::open(&fixture.0).unwrap();
        assert_eq!(w.values("value").unwrap(), [1.25, -2.5]);
        assert!(w.f32("value", &[1, 2]).is_err());
        assert!(w.integers("value").is_err());
        assert!(w.shape("missing").is_err());
        drop(w);
        let wrong = Fixture::new("unknown-v2");
        assert!(Weights::open(&wrong.0).is_err());
    }
    #[test]
    fn truncated_tensor_archive_is_rejected() {
        let fixture = Fixture::new("teamy-glados-native-v1");
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&fixture.0)
            .unwrap();
        let size = file.metadata().unwrap().len();
        file.set_len(size - 1).unwrap();
        drop(file);
        assert!(Weights::open(&fixture.0).is_err());
    }
}
