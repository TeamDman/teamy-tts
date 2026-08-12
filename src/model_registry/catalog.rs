use crate::paths::CACHE_DIR;
use crate::paths::ModelHome;
use facet::Facet;
use std::path::PathBuf;

const GLADOS_ARCHIVE_SHA256: &str =
    "afb60dd8944934ea5c67bd85de70f424c151b5f41b50dc039578716364fa68c4";
const GLADOS_ARCHIVE_SIZE_BYTES: u64 = 343_345_374;
const GLADOS_NATIVE_BUNDLE_SHA256: &str =
    "5fc80b76584ef7c078a417fb53e09fa8477b211e26458ad1ee8f4a25cf626e0f";
const GLADOS_NATIVE_BUNDLE_SIZE_BYTES: u64 = 217_016_604;

/// A model known to the teamy-tts catalog.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelDefinition {
    pub id: &'static str,
    pub display_name: &'static str,
    pub description: &'static str,
    pub revision: &'static str,
    pub sample_rate_hz: u32,
    pub voices: &'static [&'static str],
    pub archive_sha256: &'static str,
    pub archive_size_bytes: u64,
    pub native_bundle_sha256: &'static str,
    pub native_bundle_size_bytes: u64,
}

static GLADOS_VOICES: &[&str] = &["p1", "p2"];

static MODEL_CATALOG: &[ModelDefinition] = &[ModelDefinition {
    id: "glados",
    display_name: "GLaDOS",
    description: "The local GLaDOS-style ForwardTacotron and HiFiGAN pipeline.",
    revision: "glados-new",
    sample_rate_hz: 22_050,
    voices: GLADOS_VOICES,
    archive_sha256: GLADOS_ARCHIVE_SHA256,
    archive_size_bytes: GLADOS_ARCHIVE_SIZE_BYTES,
    native_bundle_sha256: GLADOS_NATIVE_BUNDLE_SHA256,
    native_bundle_size_bytes: GLADOS_NATIVE_BUNDLE_SIZE_BYTES,
}];

/// Return the immutable known-model catalog.
#[must_use]
pub fn all_models() -> &'static [ModelDefinition] {
    MODEL_CATALOG
}

/// Find a model by its stable user-facing identifier.
#[must_use]
pub fn find_model(id: &str) -> Option<ModelDefinition> {
    MODEL_CATALOG.iter().copied().find(|model| model.id == id)
}

/// The user-facing model and installation state report.
#[derive(Clone, Debug, Facet)]
pub struct ModelReport {
    pub id: String,
    pub display_name: String,
    pub description: String,
    pub revision: String,
    pub status: String,
    pub sample_rate_hz: u32,
    pub voices: Vec<String>,
    pub archive_sha256: String,
    pub archive_size_bytes: u64,
    pub native_bundle_sha256: String,
    pub native_bundle_size_bytes: u64,
    pub archive_path: String,
    pub acquisition_receipt_path: String,
    pub native_bundle_archive_path: String,
    pub native_bundle_acquisition_receipt_path: String,
    pub prepared_manifest_path: String,
}

/// Build a report without creating or mutating the model cache.
///
/// # Errors
///
/// Returns an error if the configured model root cannot be resolved.
pub fn report_for(model: ModelDefinition) -> eyre::Result<ModelReport> {
    let archive_path = raw_archive_path(model);
    let acquisition_receipt_path = acquisition_receipt_path(model);
    let native_bundle_archive_path = native_bundle_archive_path(model);
    let native_bundle_acquisition_receipt_path = native_bundle_acquisition_receipt_path(model);
    let prepared_manifest_path = prepared_manifest_path(model)?;
    let status = if prepared_manifest_path.is_file() {
        match super::inspect_prepared_model_dir(model) {
            Ok(_) => "prepared",
            Err(_) => "prepared-invalid",
        }
    } else if native_bundle_archive_path.is_file()
        && native_bundle_acquisition_receipt_path.is_file()
    {
        "native-bundle-verified"
    } else if archive_path.is_file() && acquisition_receipt_path.is_file() {
        "raw-archive-verified"
    } else if archive_path.is_file() {
        "raw-archive-present"
    } else {
        "not-acquired"
    };

    Ok(ModelReport {
        id: model.id.to_string(),
        display_name: model.display_name.to_string(),
        description: model.description.to_string(),
        revision: model.revision.to_string(),
        status: status.to_string(),
        sample_rate_hz: model.sample_rate_hz,
        voices: model
            .voices
            .iter()
            .map(|voice| (*voice).to_string())
            .collect(),
        archive_sha256: model.archive_sha256.to_string(),
        archive_size_bytes: model.archive_size_bytes,
        native_bundle_sha256: model.native_bundle_sha256.to_string(),
        native_bundle_size_bytes: model.native_bundle_size_bytes,
        archive_path: archive_path.display().to_string(),
        acquisition_receipt_path: acquisition_receipt_path.display().to_string(),
        native_bundle_archive_path: native_bundle_archive_path.display().to_string(),
        native_bundle_acquisition_receipt_path: native_bundle_acquisition_receipt_path
            .display()
            .to_string(),
        prepared_manifest_path: prepared_manifest_path.display().to_string(),
    })
}

/// Return the content-addressed raw archive path for a known model.
#[must_use]
pub fn raw_archive_path(model: ModelDefinition) -> PathBuf {
    CACHE_DIR
        .0
        .join("raw-models")
        .join(model.id)
        .join(model.archive_sha256)
        .join("models.zip")
}

/// Return the acquisition receipt path for a known model.
#[must_use]
pub fn acquisition_receipt_path(model: ModelDefinition) -> PathBuf {
    raw_archive_path(model).with_file_name("acquisition.json")
}

/// Return the content-addressed native bundle archive path for a known model.
#[must_use]
pub fn native_bundle_archive_path(model: ModelDefinition) -> PathBuf {
    CACHE_DIR
        .0
        .join("native-models")
        .join(model.id)
        .join(model.native_bundle_sha256)
        .join("native-bundle.zip")
}

/// Return the native bundle acquisition receipt path for a known model.
#[must_use]
pub fn native_bundle_acquisition_receipt_path(model: ModelDefinition) -> PathBuf {
    native_bundle_archive_path(model).with_file_name("acquisition.json")
}

/// Resolve the prepared model directory for a catalog entry.
///
/// # Errors
///
/// Returns an error if the configured model home cannot be resolved.
pub fn prepared_model_path(model: ModelDefinition) -> eyre::Result<PathBuf> {
    Ok(ModelHome::resolve()?.0.join(model.id).join(model.revision))
}

fn prepared_manifest_path(model: ModelDefinition) -> eyre::Result<PathBuf> {
    Ok(prepared_model_path(model)?.join("manifest.json"))
}
