use super::ModelDefinition;
use crate::model_registry::prepared_model_path;
use eyre::WrapErr;
use eyre::bail;
use facet::Facet;
use sha2::Digest;
use sha2::Sha256;
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use zip::ZipArchive;

pub const MANIFEST_FILE_NAME: &str = "manifest.json";

const MANIFEST_FORMAT: &str = "teamy-tts-prepared-model";
const MANIFEST_SCHEMA_VERSION: u32 = 1;
const NATIVE_BUNDLE_FILES: [&str; 6] = [
    "acoustic-model.bpk",
    "vocoder.bpk",
    "phonemizer.bpk",
    "frontend.tsv",
    "voice-p1.f32le",
    "voice-p2.f32le",
];

/// The native artifact format produced by a model preparation tool.
#[derive(Clone, Debug, Facet, PartialEq, Eq)]
pub struct PreparedModelManifest {
    pub format: String,
    pub schema_version: u32,
    pub model_id: String,
    pub revision: String,
    pub source_archive_sha256: String,
    pub source_archive_size_bytes: u64,
    pub converter_version: String,
    pub sample_rate_hz: u32,
    pub artifacts: Vec<PreparedArtifact>,
}

/// A role-specific file needed by the native TTS runtime.
#[derive(Clone, Debug, Facet, PartialEq, Eq)]
pub struct PreparedArtifact {
    pub role: String,
    pub path: String,
    pub format: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub dtype: Option<String>,
    pub shape: Option<Vec<usize>>,
}

/// A manifest plus the resolved artifact paths it describes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedModelArtifacts {
    pub root: PathBuf,
    pub manifest_path: PathBuf,
    pub manifest: PreparedModelManifest,
    pub artifact_paths: Vec<PathBuf>,
}

impl PreparedModelArtifacts {
    /// Resolve the sole artifact with a given runtime role.
    ///
    /// # Errors
    ///
    /// Returns an error when the role is absent or duplicated in the
    /// manifest.  Duplicate roles are rejected here so a runtime cannot load
    /// an artifact chosen by manifest ordering.
    pub fn path_for_role(&self, role: &str) -> eyre::Result<&Path> {
        let mut matches = self
            .manifest
            .artifacts
            .iter()
            .zip(&self.artifact_paths)
            .filter(|(artifact, _)| artifact.role == role)
            .map(|(_, path)| path.as_path());
        let Some(path) = matches.next() else {
            bail!(
                "prepared model is missing required artifact role {:?}",
                role
            );
        };
        if matches.next().is_some() {
            bail!("prepared model contains duplicate artifact role {:?}", role);
        }
        Ok(path)
    }
}

/// Prepare a native bundle into the catalog's resolved model home.
///
/// The bundle is intentionally a fixed, shallow layout.  Conversion tools
/// produce these names, and the runtime only consumes the resulting manifest:
///
/// - `acoustic-model.bpk`
/// - `vocoder.bpk`
/// - `frontend.tsv`
/// - `phonemizer.bpk`
/// - `voice-p1.f32le`
/// - `voice-p2.f32le`
///
/// # Errors
///
/// Returns an error when a required bundle file is missing, the destination
/// already exists without `overwrite`, or the installed artifacts fail their
/// manifest verification.
#[expect(
    clippy::too_many_lines,
    reason = "The fixed bundle mapping and atomic install are one auditable preparation operation."
)]
pub fn prepare_native_bundle(
    model: ModelDefinition,
    source_dir: &Path,
    overwrite: bool,
) -> eyre::Result<PreparedModelArtifacts> {
    if !source_dir.is_dir() {
        bail!(
            "native model bundle is not a directory: {}",
            source_dir.display()
        );
    }
    let root = prepared_model_path(model)?;
    if root.exists() && !overwrite {
        bail!(
            "prepared model destination already exists: {}; pass --force to update it",
            root.display()
        );
    }
    let parent = root
        .parent()
        .ok_or_else(|| eyre::eyre!("prepared model path has no parent: {}", root.display()))?;
    std::fs::create_dir_all(parent).wrap_err_with(|| {
        format!(
            "failed to create prepared model parent {}",
            parent.display()
        )
    })?;
    let stage = parent.join(format!(
        ".{}.staging-{}",
        root.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model"),
        std::process::id()
    ));
    if stage.exists() {
        bail!(
            "prepared model staging directory already exists: {}; remove the abandoned staging directory after checking it",
            stage.display()
        );
    }
    std::fs::create_dir_all(&stage).wrap_err_with(|| {
        format!(
            "failed to create prepared model staging directory {}",
            stage.display()
        )
    })?;

    let definitions = [
        (
            "acoustic-model",
            "acoustic-model.bpk",
            "burnpack",
            None,
            None,
        ),
        ("vocoder", "vocoder.bpk", "burnpack", None, None),
        (
            "frontend-dictionary",
            "frontend.tsv",
            "utf8-tsv",
            None,
            None,
        ),
        (
            "frontend-phonemizer",
            "phonemizer.bpk",
            "burnpack",
            None,
            None,
        ),
        (
            "voice-p1",
            "voice-p1.f32le",
            "float32-le",
            Some("f32"),
            Some(vec![1, 256]),
        ),
        (
            "voice-p2",
            "voice-p2.f32le",
            "float32-le",
            Some("f32"),
            Some(vec![1, 256]),
        ),
    ];
    let mut artifacts = Vec::with_capacity(definitions.len());
    for (role, source_name, format, dtype, shape) in definitions {
        let source = source_dir.join(source_name);
        let metadata = std::fs::metadata(&source)
            .wrap_err_with(|| format!("failed to inspect bundle artifact {}", source.display()))?;
        if !metadata.is_file() {
            bail!("native bundle artifact is not a file: {}", source.display());
        }
        let target = stage.join(source_name);
        std::fs::copy(&source, &target).wrap_err_with(|| {
            format!(
                "failed to copy native bundle artifact {} to {}",
                source.display(),
                target.display()
            )
        })?;
        artifacts.push(PreparedArtifact {
            role: role.to_string(),
            path: source_name.to_string(),
            format: format.to_string(),
            sha256: sha256_file(&target)?,
            size_bytes: metadata.len(),
            dtype: dtype.map(str::to_string),
            shape,
        });
    }

    let manifest = PreparedModelManifest {
        format: MANIFEST_FORMAT.to_string(),
        schema_version: MANIFEST_SCHEMA_VERSION,
        model_id: model.id.to_string(),
        revision: model.revision.to_string(),
        source_archive_sha256: model.archive_sha256.to_string(),
        source_archive_size_bytes: model.archive_size_bytes,
        converter_version: "native-bundle-v1".to_string(),
        sample_rate_hz: model.sample_rate_hz,
        artifacts,
    };
    let manifest_path = stage.join(MANIFEST_FILE_NAME);
    let manifest_contents = facet_json::to_string_pretty(&manifest)
        .wrap_err("failed to serialize prepared model manifest")?;
    std::fs::write(&manifest_path, manifest_contents).wrap_err_with(|| {
        format!(
            "failed to write prepared model manifest {}",
            manifest_path.display()
        )
    })?;

    let staged = inspect_prepared_model_root(model, &stage)?;
    verify_prepared_model_artifacts(&staged)?;
    if root.exists() {
        let backup = parent.join(format!(
            ".{}.previous-{}",
            root.file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("model"),
            std::process::id()
        ));
        if backup.exists() {
            bail!(
                "prepared model backup already exists: {}; refusing to overwrite it",
                backup.display()
            );
        }
        std::fs::rename(&root, &backup).wrap_err_with(|| {
            format!(
                "failed to move the previous prepared model to {}",
                backup.display()
            )
        })?;
    }
    std::fs::rename(&stage, &root).wrap_err_with(|| {
        format!(
            "failed to atomically install prepared model from {} to {}",
            stage.display(),
            root.display()
        )
    })?;
    let installed = inspect_prepared_model_root(model, &root)?;
    verify_prepared_model_artifacts(&installed)?;
    Ok(installed)
}

/// Extract and prepare a fixed native bundle archive without invoking an
/// external archive tool.
///
/// Only the six exact root-level runtime artifacts are extracted. Unknown
/// entries are ignored, and archive paths never become filesystem paths, so a
/// malformed archive cannot write outside the temporary extraction directory.
///
/// # Errors
///
/// Returns an error if the archive is unreadable, a required artifact is
/// absent or duplicated, extraction fails, or preparation fails.
pub fn prepare_native_bundle_archive(
    model: ModelDefinition,
    source_archive: &Path,
    overwrite: bool,
) -> eyre::Result<PreparedModelArtifacts> {
    if !source_archive.is_file() {
        bail!(
            "native model bundle archive is not a file: {}",
            source_archive.display()
        );
    }
    let metadata = std::fs::metadata(source_archive).wrap_err_with(|| {
        format!(
            "failed to inspect native model bundle archive {}",
            source_archive.display()
        )
    })?;
    if metadata.len() != model.native_bundle_size_bytes {
        bail!(
            "native model bundle archive size mismatch: expected {}, received {}",
            model.native_bundle_size_bytes,
            metadata.len()
        );
    }
    let actual_sha256 = sha256_file(source_archive)?;
    if actual_sha256 != model.native_bundle_sha256 {
        bail!(
            "native model bundle archive SHA-256 mismatch: expected {}, received {}",
            model.native_bundle_sha256,
            actual_sha256
        );
    }
    let parent = source_archive.parent().unwrap_or_else(|| Path::new("."));
    let extraction_dir = parent.join(format!(
        ".teamy-tts-{}-bundle-{}",
        model.id,
        std::process::id()
    ));
    extract_native_bundle_archive(source_archive, &extraction_dir)?;
    let result = prepare_native_bundle(model, &extraction_dir, overwrite);
    let cleanup_result = std::fs::remove_dir_all(&extraction_dir).wrap_err_with(|| {
        format!(
            "failed to remove temporary bundle {}",
            extraction_dir.display()
        )
    });
    match (result, cleanup_result) {
        (Ok(artifacts), Ok(())) => Ok(artifacts),
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
        (Err(error), Err(cleanup_error)) => Err(error.wrap_err(cleanup_error)),
    }
}

/// Extract the fixed native bundle files into a new directory.
///
/// This helper is public so callers can rehearse extraction separately from
/// model installation. The destination must not already exist.
///
/// # Errors
///
/// Returns an error if the archive is malformed, unsafe, incomplete, or the
/// destination cannot be installed atomically.
pub fn extract_native_bundle_archive(
    source_archive: &Path,
    destination: &Path,
) -> eyre::Result<()> {
    if destination.exists() {
        bail!(
            "native bundle extraction destination already exists: {}",
            destination.display()
        );
    }
    let parent = destination
        .parent()
        .ok_or_else(|| eyre::eyre!("native bundle destination has no parent"))?;
    std::fs::create_dir_all(parent).wrap_err_with(|| {
        format!(
            "failed to create native bundle extraction parent {}",
            parent.display()
        )
    })?;
    let stage = parent.join(format!(
        ".{}-staging-{}",
        destination
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("native-bundle"),
        std::process::id()
    ));
    if stage.exists() {
        bail!(
            "native bundle extraction staging directory already exists: {}",
            stage.display()
        );
    }
    std::fs::create_dir_all(&stage).wrap_err_with(|| {
        format!(
            "failed to create native bundle extraction staging directory {}",
            stage.display()
        )
    })?;

    let result = extract_native_bundle_archive_to_stage(source_archive, &stage);
    if let Err(error) = result {
        let _ = std::fs::remove_dir_all(&stage);
        return Err(error);
    }
    std::fs::rename(&stage, destination).wrap_err_with(|| {
        format!(
            "failed to atomically install native bundle extraction at {}",
            destination.display()
        )
    })
}

fn extract_native_bundle_archive_to_stage(source_archive: &Path, stage: &Path) -> eyre::Result<()> {
    let file = File::open(source_archive).wrap_err_with(|| {
        format!(
            "failed to open native bundle archive {}",
            source_archive.display()
        )
    })?;
    let mut archive = ZipArchive::new(file).wrap_err_with(|| {
        format!(
            "failed to read native bundle ZIP archive {}",
            source_archive.display()
        )
    })?;
    let mut found = [false; NATIVE_BUNDLE_FILES.len()];
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .wrap_err("failed to inspect native bundle ZIP entry")?;
        let Some(file_index) = NATIVE_BUNDLE_FILES
            .iter()
            .position(|name| *name == entry.name())
        else {
            continue;
        };
        if found[file_index] {
            bail!(
                "native bundle archive contains duplicate entry {:?}",
                entry.name()
            );
        }
        if entry.is_dir() {
            bail!(
                "native bundle archive entry {:?} is a directory",
                entry.name()
            );
        }
        let target = stage.join(NATIVE_BUNDLE_FILES[file_index]);
        let mut output = File::create(&target).wrap_err_with(|| {
            format!("failed to create extracted artifact {}", target.display())
        })?;
        std::io::copy(&mut entry, &mut output).wrap_err_with(|| {
            format!("failed to extract native bundle artifact {}", entry.name())
        })?;
        output
            .sync_all()
            .wrap_err_with(|| format!("failed to sync extracted artifact {}", target.display()))?;
        found[file_index] = true;
    }
    if let Some(missing) = NATIVE_BUNDLE_FILES
        .iter()
        .zip(found)
        .find_map(|(name, present)| (!present).then_some(*name))
    {
        bail!(
            "native bundle archive is missing required entry {:?}",
            missing
        );
    }
    Ok(())
}

/// Inspect the native prepared-model layout without hashing large weight files.
///
/// The preparation step performs full hashing. Later runtime startup only needs
/// to prove that the manifest is coherent and that its files have not been
/// truncated or replaced with files of a different length.
///
/// # Errors
///
/// Returns an error if the manifest is absent, malformed, mismatched with the
/// catalog, or refers to missing or unsafe paths.
pub fn inspect_prepared_model_dir(model: ModelDefinition) -> eyre::Result<PreparedModelArtifacts> {
    let root = prepared_model_path(model)?;
    inspect_prepared_model_root(model, &root)
}

/// Inspect a native prepared-model directory at an explicit path.
///
/// This is primarily useful to preparation tools and tests that need to work
/// with a staging directory instead of the user's configured model home.
///
/// # Errors
///
/// Returns an error if the manifest or any described artifact is invalid.
pub fn inspect_prepared_model_root(
    model: ModelDefinition,
    root: &Path,
) -> eyre::Result<PreparedModelArtifacts> {
    if !root.is_dir() {
        bail!("prepared model directory is missing: {}", root.display());
    }

    let manifest_path = root.join(MANIFEST_FILE_NAME);
    let manifest_contents = std::fs::read_to_string(&manifest_path)
        .wrap_err_with(|| format!("failed to read {}", manifest_path.display()))?;
    let manifest: PreparedModelManifest = facet_json::from_str(&manifest_contents)
        .wrap_err_with(|| format!("failed to parse {}", manifest_path.display()))?;

    validate_manifest(model, &manifest)?;

    let mut artifact_paths = Vec::with_capacity(manifest.artifacts.len());
    for artifact in &manifest.artifacts {
        let relative_path = Path::new(&artifact.path);
        validate_relative_artifact_path(relative_path, artifact)?;
        let artifact_path = root.join(relative_path);
        let metadata = std::fs::metadata(&artifact_path)
            .wrap_err_with(|| format!("failed to inspect {}", artifact_path.display()))?;
        if !metadata.is_file() {
            bail!(
                "prepared artifact is not a file: {}",
                artifact_path.display()
            );
        }
        if metadata.len() != artifact.size_bytes {
            bail!(
                "prepared artifact {} has {} bytes, expected {}",
                artifact_path.display(),
                metadata.len(),
                artifact.size_bytes
            );
        }
        artifact_paths.push(artifact_path);
    }

    Ok(PreparedModelArtifacts {
        root: root.to_path_buf(),
        manifest_path,
        manifest,
        artifact_paths,
    })
}

/// Verify every native artifact against the hash recorded in its manifest.
///
/// # Errors
///
/// Returns an error if the layout is incomplete or any artifact hash differs.
pub fn verify_prepared_model_dir(model: ModelDefinition) -> eyre::Result<PreparedModelArtifacts> {
    let artifacts = inspect_prepared_model_dir(model)?;
    verify_prepared_model_artifacts(&artifacts)?;
    Ok(artifacts)
}

/// Verify an already inspected native model directory.
///
/// # Errors
///
/// Returns an error if an artifact cannot be read or its SHA-256 differs from
/// the manifest.
pub fn verify_prepared_model_artifacts(artifacts: &PreparedModelArtifacts) -> eyre::Result<()> {
    for (artifact, artifact_path) in artifacts
        .manifest
        .artifacts
        .iter()
        .zip(&artifacts.artifact_paths)
    {
        let actual_sha256 = sha256_file(artifact_path)?;
        if actual_sha256 != artifact.sha256 {
            bail!(
                "prepared artifact {} has SHA-256 {}, expected {}",
                artifact_path.display(),
                actual_sha256,
                artifact.sha256
            );
        }
    }
    Ok(())
}

fn validate_manifest(model: ModelDefinition, manifest: &PreparedModelManifest) -> eyre::Result<()> {
    if manifest.format != MANIFEST_FORMAT {
        bail!(
            "prepared model manifest has unsupported format {:?}; expected {:?}",
            manifest.format,
            MANIFEST_FORMAT
        );
    }
    if manifest.schema_version != MANIFEST_SCHEMA_VERSION {
        bail!(
            "prepared model manifest has unsupported schema version {}; expected {}",
            manifest.schema_version,
            MANIFEST_SCHEMA_VERSION
        );
    }
    if manifest.model_id != model.id {
        bail!(
            "prepared model manifest identifies {:?}; expected {:?}",
            manifest.model_id,
            model.id
        );
    }
    if manifest.revision != model.revision {
        bail!(
            "prepared model manifest has revision {:?}; expected {:?}",
            manifest.revision,
            model.revision
        );
    }
    if manifest.source_archive_sha256 != model.archive_sha256 {
        bail!(
            "prepared model manifest refers to archive SHA-256 {}, expected {}",
            manifest.source_archive_sha256,
            model.archive_sha256
        );
    }
    if manifest.source_archive_size_bytes != model.archive_size_bytes {
        bail!(
            "prepared model manifest refers to archive size {}, expected {}",
            manifest.source_archive_size_bytes,
            model.archive_size_bytes
        );
    }
    if manifest.sample_rate_hz != model.sample_rate_hz {
        bail!(
            "prepared model manifest has sample rate {}, expected {}",
            manifest.sample_rate_hz,
            model.sample_rate_hz
        );
    }
    if manifest.artifacts.is_empty() {
        bail!("prepared model manifest does not contain any artifacts");
    }
    Ok(())
}

fn validate_relative_artifact_path(path: &Path, artifact: &PreparedArtifact) -> eyre::Result<()> {
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                std::path::Component::ParentDir | std::path::Component::Prefix(_)
            )
        })
    {
        bail!(
            "prepared artifact role {:?} has unsafe relative path {:?}",
            artifact.role,
            artifact.path
        );
    }
    if artifact.role.is_empty() || artifact.format.is_empty() || artifact.sha256.is_empty() {
        bail!(
            "prepared artifact {:?} must include role, format, and SHA-256",
            artifact.path
        );
    }
    if path.as_os_str().is_empty() {
        bail!("prepared artifact {:?} has an empty path", artifact.role);
    }
    Ok(())
}

fn sha256_file(path: &Path) -> eyre::Result<String> {
    let mut file = std::fs::File::open(path)
        .wrap_err_with(|| format!("failed to open prepared artifact {}", path.display()))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)
        .wrap_err_with(|| format!("failed to hash prepared artifact {}", path.display()))?;
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::time::SystemTime;
    use std::time::UNIX_EPOCH;
    use zip::CompressionMethod;
    use zip::write::SimpleFileOptions;
    use zip::write::ZipWriter;

    fn test_model() -> ModelDefinition {
        super::super::find_model("glados").expect("test model should exist")
    }

    fn unique_temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("teamy-tts-{name}-{nanos}"))
    }

    #[test]
    fn inspect_and_verify_prepared_manifest() {
        let model = test_model();
        let root = unique_temp_dir("prepared-model");
        std::fs::create_dir_all(&root).expect("test root should be creatable");
        let contents = b"native test artifact";
        let artifact_path = root.join("acoustic/model.bpk");
        std::fs::create_dir_all(
            artifact_path
                .parent()
                .expect("artifact parent should exist"),
        )
        .expect("artifact parent should be creatable");
        std::fs::write(&artifact_path, contents).expect("test artifact should be writable");

        let mut hasher = Sha256::new();
        hasher.update(contents);
        let manifest = PreparedModelManifest {
            format: MANIFEST_FORMAT.to_string(),
            schema_version: MANIFEST_SCHEMA_VERSION,
            model_id: model.id.to_string(),
            revision: model.revision.to_string(),
            source_archive_sha256: model.archive_sha256.to_string(),
            source_archive_size_bytes: model.archive_size_bytes,
            converter_version: "test".to_string(),
            sample_rate_hz: model.sample_rate_hz,
            artifacts: vec![PreparedArtifact {
                role: "acoustic-model".to_string(),
                path: "acoustic/model.bpk".to_string(),
                format: "burnpack".to_string(),
                sha256: format!("{:x}", hasher.finalize()),
                size_bytes: contents.len() as u64,
                dtype: None,
                shape: None,
            }],
        };
        std::fs::write(
            root.join(MANIFEST_FILE_NAME),
            facet_json::to_string_pretty(&manifest)
                .expect("manifest should serialize")
                .as_bytes(),
        )
        .expect("manifest should be writable");

        let artifacts = inspect_prepared_model_root(model, &root)
            .expect("prepared model should be inspectable");
        verify_prepared_model_artifacts(&artifacts).expect("prepared model should verify");

        std::fs::remove_dir_all(root).expect("test root should be removable");
    }

    #[test]
    fn reject_parent_directory_artifact_path() {
        let model = test_model();
        let root = unique_temp_dir("unsafe-prepared-model");
        std::fs::create_dir_all(&root).expect("test root should be creatable");
        let manifest = PreparedModelManifest {
            format: MANIFEST_FORMAT.to_string(),
            schema_version: MANIFEST_SCHEMA_VERSION,
            model_id: model.id.to_string(),
            revision: model.revision.to_string(),
            source_archive_sha256: model.archive_sha256.to_string(),
            source_archive_size_bytes: model.archive_size_bytes,
            converter_version: "test".to_string(),
            sample_rate_hz: model.sample_rate_hz,
            artifacts: vec![PreparedArtifact {
                role: "acoustic-model".to_string(),
                path: "../outside.bpk".to_string(),
                format: "burnpack".to_string(),
                sha256: "00".repeat(32),
                size_bytes: 0,
                dtype: None,
                shape: None,
            }],
        };
        std::fs::write(
            root.join(MANIFEST_FILE_NAME),
            facet_json::to_string(&manifest)
                .expect("manifest should serialize")
                .as_bytes(),
        )
        .expect("manifest should be writable");

        let error = inspect_prepared_model_root(model, &root)
            .expect_err("unsafe artifact paths should be rejected");
        assert!(error.to_string().contains("unsafe relative path"));

        std::fs::remove_dir_all(root).expect("test root should be removable");
    }

    #[test]
    fn extract_native_bundle_zip_uses_only_fixed_root_entries() {
        let root = unique_temp_dir("native-bundle-zip");
        std::fs::create_dir_all(&root).expect("test root should be creatable");
        let archive_path = root.join("bundle.zip");
        let file = std::fs::File::create(&archive_path).expect("archive should be writable");
        let mut writer = ZipWriter::new(file);
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
        for name in NATIVE_BUNDLE_FILES {
            writer
                .start_file(name, options)
                .expect("bundle entry should be creatable");
            writer
                .write_all(name.as_bytes())
                .expect("bundle entry should be writable");
        }
        writer
            .start_file("../outside.txt", options)
            .expect("unknown entry should be creatable");
        writer
            .write_all(b"must not be extracted")
            .expect("unknown entry should be writable");
        writer.finish().expect("archive should be finishable");

        let destination = root.join("extracted");
        extract_native_bundle_archive(&archive_path, &destination)
            .expect("native bundle archive should extract");
        for name in NATIVE_BUNDLE_FILES {
            assert_eq!(
                std::fs::read_to_string(destination.join(name))
                    .expect("required entry should be extracted"),
                name
            );
        }
        assert!(!root.join("outside.txt").exists());
        std::fs::remove_dir_all(root).expect("test root should be removable");
    }
}
