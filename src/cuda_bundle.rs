//! Verified runtime-only CUDA model acquisition. Legacy archives use their own path.
use eyre::{Context, Result, ensure};
use facet::Facet;
use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use teamy_cancellation::CancellationToken;
use tokio::io::AsyncWriteExt;

pub const SOURCE_ENV: &str = "TEAMY_TTS_TEAMY_CUDA_SOURCE_URL";
const PUBLIC_ROOT: &str = "https://pub-efc9d45264d54fffb27e33d408633ea8.r2.dev";
const SCHEMA: &str = "teamy-glados-native-v1";

#[derive(Debug, Facet, PartialEq, Eq)]
pub struct FileRecord {
    pub bytes: u64,
    pub sha256: String,
}
#[derive(Debug, Facet)]
pub struct Catalog {
    pub schema: String,
    pub archive_bytes: u64,
    pub archive_sha256: String,
    pub sample_rate: u32,
    pub files: BTreeMap<String, FileRecord>,
}
#[derive(Debug, Facet)]
struct Manifest {
    schema: String,
    sample_rate: u32,
    precision: String,
    voices: Vec<String>,
    files: BTreeMap<String, FileRecord>,
}
#[derive(Debug, Facet)]
pub struct Acquisition {
    pub model: String,
    pub format: String,
    pub archive_path: String,
    pub prepared_dir: String,
    pub bytes: u64,
    pub sha256: String,
    pub verified: bool,
    pub configured: bool,
}

pub fn catalog() -> Result<Catalog> {
    Ok(facet_json::from_str(include_str!(
        "../assets/glados-native-v1.json"
    ))?)
}
pub fn source_url() -> Result<String> {
    match std::env::var(SOURCE_ENV) {
        Ok(value) => {
            ensure!(!value.trim().is_empty(), "{SOURCE_ENV} cannot be empty");
            Ok(value)
        }
        Err(std::env::VarError::NotPresent) => Ok(format!(
            "{PUBLIC_ROOT}/cuda-native/glados/v1/{}/model.zip",
            catalog()?.archive_sha256
        )),
        Err(error) => Err(error.into()),
    }
}
pub fn installed_path() -> Result<PathBuf> {
    Ok(crate::paths::ModelHome::resolve()?
        .0
        .join("glados")
        .join(SCHEMA)
        .join(catalog()?.archive_sha256))
}
pub fn archive_path() -> Result<PathBuf> {
    Ok(crate::paths::CacheHome::resolve()?
        .0
        .join("cuda-native-models")
        .join(catalog()?.archive_sha256)
        .join("model.zip"))
}

fn hash_file(path: &Path) -> Result<(u64, String)> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut bytes = 0u64;
    let mut buffer = [0u8; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        bytes += n as u64;
        hasher.update(&buffer[..n]);
    }
    Ok((bytes, format!("{:x}", hasher.finalize())))
}
fn verify_file(path: &Path, record: &FileRecord) -> Result<()> {
    ensure!(
        path.metadata()?.len() == record.bytes,
        "{} size mismatch",
        path.display()
    );
    ensure!(
        hash_file(path)?.1 == record.sha256,
        "{} SHA-256 mismatch",
        path.display()
    );
    Ok(())
}
pub fn verify_dir(path: &Path) -> Result<()> {
    let catalog = catalog()?;
    for (name, expected) in &catalog.files {
        verify_file(&path.join(name), expected)?;
    }
    let manifest: Manifest =
        facet_json::from_str(&fs::read_to_string(path.join("manifest.json"))?)?;
    ensure!(
        manifest.schema == SCHEMA
            && manifest.sample_rate == 22050
            && manifest.precision == "float32"
            && manifest.voices == ["p1", "p2"],
        "unsupported CUDA runtime manifest"
    );
    for (name, record) in &manifest.files {
        ensure!(
            catalog.files.get(name) == Some(record),
            "runtime manifest does not match the pinned catalog"
        );
    }
    ensure!(
        manifest.files.len() == 2,
        "runtime manifest must name weights and frontend"
    );
    Ok(())
}

struct Staging(PathBuf);
impl Staging {
    fn new(parent: &Path) -> Result<Self> {
        fs::create_dir_all(parent)?;
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = parent.join(format!(".teamy-staging-{}-{nonce}", std::process::id()));
        fs::create_dir(&path)?;
        Ok(Self(path))
    }
}
impl Drop for Staging {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

pub fn install_archive(archive: &Path, destination: &Path) -> Result<()> {
    let catalog = catalog()?;
    verify_file(
        archive,
        &FileRecord {
            bytes: catalog.archive_bytes,
            sha256: catalog.archive_sha256,
        },
    )?;
    if destination.exists() {
        verify_dir(destination)?;
        return Ok(());
    }
    let parent = destination
        .parent()
        .ok_or_else(|| eyre::eyre!("model destination needs a parent"))?;
    let staging = Staging::new(parent)?;
    let mut zip = zip::ZipArchive::new(File::open(archive)?)?;
    ensure!(
        zip.len() == catalog.files.len(),
        "CUDA bundle must contain exactly three runtime files"
    );
    let mut seen = BTreeSet::new();
    for index in 0..zip.len() {
        let mut entry = zip.by_index(index)?;
        let name = entry.name().to_owned();
        let expected = catalog
            .files
            .get(&name)
            .ok_or_else(|| eyre::eyre!("unexpected CUDA bundle entry"))?;
        ensure!(
            seen.insert(name.clone()) && entry.is_file() && entry.size() == expected.bytes,
            "duplicate or invalid CUDA bundle entry"
        );
        ensure!(
            entry
                .unix_mode()
                .is_none_or(|mode| mode & 0o170000 == 0 || mode & 0o170000 == 0o100000),
            "CUDA bundle entries must be regular files"
        );
        let output = staging.0.join(&name); // name came from the fixed root-file allowlist.
        let mut file = File::options().write(true).create_new(true).open(&output)?;
        let mut written = 0u64;
        let mut buffer = [0u8; 65536];
        loop {
            let n = entry.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            written += n as u64;
            ensure!(
                written <= expected.bytes,
                "CUDA bundle entry exceeds its declared size"
            );
            file.write_all(&buffer[..n])?;
        }
        file.sync_all()?;
        drop(file);
        verify_file(&output, expected)?;
    }
    verify_dir(&staging.0)?;
    match fs::rename(&staging.0, destination) {
        Ok(()) => Ok(()),
        Err(_) if destination.exists() => verify_dir(destination),
        Err(error) => Err(error).wrap_err("install verified CUDA model directory"),
    }
}

pub fn prepare(source_dir: Option<&Path>, source_archive: Option<&Path>) -> Result<Acquisition> {
    ensure!(
        source_dir.is_none() || source_archive.is_none(),
        "choose only one source directory or archive"
    );
    let destination = installed_path()?;
    let catalog = catalog()?;
    let archive = source_archive
        .map(Path::to_path_buf)
        .unwrap_or(archive_path()?);
    if let Some(source) = source_dir {
        let staging = Staging::new(destination.parent().unwrap())?;
        for name in ["weights.safetensors", "frontend.tsv"] {
            verify_file(&source.join(name), &catalog.files[name])?;
            fs::copy(source.join(name), staging.0.join(name))?;
        }
        fs::write(
            staging.0.join("manifest.json"),
            include_bytes!("../assets/glados-native-v1-manifest.json"),
        )?;
        verify_dir(&staging.0)?;
        if destination.exists() {
            verify_dir(&destination)?;
        } else {
            fs::rename(&staging.0, &destination)?;
        }
    } else {
        install_archive(&archive, &destination)?;
    }
    finish(&archive, &destination, catalog)
}

fn finish(archive: &Path, destination: &Path, catalog: Catalog) -> Result<Acquisition> {
    let mut config = crate::config::load()?;
    let configured = config.native_model_dir.is_none();
    if configured {
        config.native_model_dir = Some(destination.display().to_string());
        crate::config::save(&config)?;
    }
    Ok(Acquisition {
        model: "glados".into(),
        format: SCHEMA.into(),
        archive_path: archive.display().to_string(),
        prepared_dir: destination.display().to_string(),
        bytes: catalog.archive_bytes,
        sha256: catalog.archive_sha256,
        verified: true,
        configured,
    })
}

pub async fn acquire(source: &str, cancellation: CancellationToken) -> Result<Acquisition> {
    ensure!(
        source.eq_ignore_ascii_case("teamy"),
        "CUDA-native weights currently use source Teamy; the legacy backend retains its other sources"
    );
    let catalog = catalog()?;
    let destination = installed_path()?;
    let archive = archive_path()?;
    if !destination.exists() {
        let expected = FileRecord {
            bytes: catalog.archive_bytes,
            sha256: catalog.archive_sha256.clone(),
        };
        if !archive.is_file() || verify_file(&archive, &expected).is_err() {
            let staging = Staging::new(archive.parent().unwrap())?;
            let partial = staging.0.join("model.zip");
            let url = source_url()?;
            let client = reqwest::Client::builder()
                .user_agent(concat!("teamy-tts/", env!("CARGO_PKG_VERSION")))
                .connect_timeout(Duration::from_secs(15))
                .timeout(Duration::from_secs(300))
                .build()?;
            // Do not put an override URL (which could be signed) in errors or receipts.
            let response = client
                .get(&url)
                .send()
                .await
                .map_err(|e| eyre::eyre!("CUDA bundle request failed: {}", e.without_url()))?;
            ensure!(
                response.status().is_success(),
                "CUDA bundle source returned HTTP {}",
                response.status()
            );
            ensure!(
                response
                    .content_length()
                    .is_none_or(|len| len == catalog.archive_bytes),
                "CUDA bundle HTTP size mismatch"
            );
            let mut stream = response.bytes_stream();
            let mut file = tokio::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&partial)
                .await?;
            let mut received = 0u64;
            let mut hasher = Sha256::new();
            while let Some(chunk) = stream.next().await {
                cancellation.bail_if_cancelled()?;
                let chunk = chunk
                    .map_err(|e| eyre::eyre!("CUDA bundle stream failed: {}", e.without_url()))?;
                received += chunk.len() as u64;
                ensure!(
                    received <= catalog.archive_bytes,
                    "CUDA bundle download exceeds its pinned size"
                );
                hasher.update(&chunk);
                file.write_all(&chunk).await?;
            }
            file.sync_all().await?;
            drop(file);
            ensure!(
                received == catalog.archive_bytes
                    && format!("{:x}", hasher.finalize()) == catalog.archive_sha256,
                "CUDA bundle size or SHA-256 mismatch"
            );
            // A verified pre-existing immutable archive can be reused if another installer won.
            if !archive.exists() || verify_file(&archive, &expected).is_err() {
                tokio::fs::rename(&partial, &archive).await?;
            }
        }
        cancellation.bail_if_cancelled()?;
        install_archive(&archive, &destination)?;
    } else {
        verify_dir(&destination)?;
    }
    // Preserve an explicit model selection; configure a first install automatically.
    finish(&archive, &destination, catalog)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pinned_catalog_names_only_runtime_files() {
        let catalog = catalog().unwrap();
        assert_eq!(catalog.schema, SCHEMA);
        assert_eq!(
            catalog.files.keys().map(String::as_str).collect::<Vec<_>>(),
            ["frontend.tsv", "manifest.json", "weights.safetensors"]
        );
        assert_eq!(catalog.sample_rate, 22050);
        assert!(catalog.archive_bytes > 200_000_000);
    }
}
