mod app_home;
mod cache;
mod model;

pub use app_home::*;
pub use cache::*;
pub use model::*;

pub const APP_HOME_ENV_VAR: &str = "TEAMY_TTS_HOME_DIR";
pub const APP_HOME_DIR_NAME: &str = "teamy-tts";

pub const APP_CACHE_ENV_VAR: &str = "TEAMY_TTS_CACHE_DIR";
pub const APP_CACHE_DIR_NAME: &str = "teamy-tts";
