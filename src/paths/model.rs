use super::CacheHome;
pub use crate::config::MODEL_DIR_ENV_VAR;
use std::ops::Deref;
use std::path::Path;
use std::path::PathBuf;

/// Resolved root for prepared model revisions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelHome(pub PathBuf);

impl ModelHome {
    /// Resolve the model root without creating or modifying it.
    ///
    /// # Errors
    ///
    /// Returns an error when the cache directory cannot be resolved or the
    /// override is empty.
    pub fn resolve() -> eyre::Result<Self> {
        if let Some(override_dir) = crate::config::effective_model_dir()? {
            return Ok(Self(override_dir));
        }

        Ok(Self(CacheHome::resolve()?.0.join("models")))
    }
}

impl Deref for ModelHome {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        self.0.as_path()
    }
}
