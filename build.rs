use sha2::Digest;
use sha2::Sha256;
use std::process::Command;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

fn main() {
    add_build_script_inputs();
    add_source_fingerprint();
    add_exe_resources();
    add_git_metadata();
    add_build_timestamp();
    add_torchscript_bridge();
    add_vulkan_shaders();
    add_windows_main_stack();
}

/// Re-run the build script when normal binary inputs change so embedded build metadata stays fresh.
fn add_build_script_inputs() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=Cargo.lock");
    println!("cargo:rerun-if-changed=src");
}

/// Give benchmark receipts a content identity while the worktree is dirty.
///
/// Git's ordinary dirty bit is intentionally coarse: changing one shader or
/// backend implementation leaves the same `dirty` marker. The receipt cache
/// must distinguish those builds, so hash the inputs that can affect the
/// executable or embedded Vulkan shaders in a stable path order.
fn add_source_fingerprint() {
    let mut files = Vec::new();
    for root in [
        std::path::Path::new("src"),
        std::path::Path::new("resources/vulkan"),
        std::path::Path::new("build.rs"),
        std::path::Path::new("Cargo.toml"),
        std::path::Path::new("Cargo.lock"),
    ] {
        collect_files(root, &mut files);
    }
    files.sort();

    let mut digest = Sha256::new();
    for path in files {
        let path_string = path.to_string_lossy().replace('\\', "/");
        digest.update(path_string.as_bytes());
        digest.update([0]);
        digest.update(
            std::fs::read(&path)
                .unwrap_or_else(|error| panic!("failed to hash {}: {error}", path.display())),
        );
        digest.update([0]);
    }
    println!(
        "cargo:rustc-env=TEAMY_TTS_SOURCE_FINGERPRINT={:x}",
        digest.finalize()
    );
}

fn collect_files(path: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
    if path.is_file() {
        files.push(path.to_path_buf());
        return;
    }
    if !path.is_dir() {
        return;
    }
    let entries = std::fs::read_dir(path)
        .unwrap_or_else(|error| panic!("failed to enumerate {}: {error}", path.display()));
    for entry in entries {
        let entry =
            entry.unwrap_or_else(|error| panic!("failed to enumerate {}: {error}", path.display()));
        collect_files(&entry.path(), files);
    }
}

/// Build the optional `LibTorch` C++ bridge against an explicitly selected
/// `LibTorch` installation. The default Burn build never needs a C++ toolchain.
fn add_torchscript_bridge() {
    if std::env::var_os("CARGO_FEATURE_TORCHSCRIPT").is_none() {
        return;
    }

    let libtorch = std::env::var_os("LIBTORCH")
        .map(std::path::PathBuf::from)
        .expect("LIBTORCH must point to a LibTorch/PyTorch installation when the torchscript feature is enabled");
    let include = libtorch.join("include");
    let api_include = include
        .join("torch")
        .join("csrc")
        .join("api")
        .join("include");
    let lib = libtorch.join("lib");
    assert!(
        include.is_dir(),
        "LIBTORCH include directory does not exist: {}",
        include.display()
    );
    assert!(
        lib.is_dir(),
        "LIBTORCH lib directory does not exist: {}",
        lib.display()
    );

    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-changed=src/native_glados/torchscript_bridge.cpp");
    cc::Build::new()
        .cpp(true)
        .file("src/native_glados/torchscript_bridge.cpp")
        .include(&include)
        .include(&api_include)
        .flag_if_supported("/std:c++17")
        .flag_if_supported("/EHsc")
        .compile("teamy_tts_torchscript_bridge");

    println!("cargo:rustc-link-search=native={}", lib.display());
    for library in ["torch", "torch_cpu", "torch_cuda", "c10_cuda", "c10"] {
        println!("cargo:rustc-link-lib=dylib={library}");
    }
}

/// Compile the small Vulkan compute probe shader when the optional feature is
/// enabled. The compiler can be supplied through GLSLC, `VULKAN_SDK`, or PATH.
#[expect(
    clippy::too_many_lines,
    reason = "The feature-gated shader manifest keeps all embedded Vulkan inputs auditable."
)]
fn add_vulkan_shaders() {
    if std::env::var_os("CARGO_FEATURE_VULKAN").is_none() {
        return;
    }

    let vector_add_output = compile_vulkan_shader(
        std::path::Path::new("resources/vulkan/vector_add.comp"),
        "teamy_tts_vector_add.comp.spv",
    );
    let matrix_multiply_output = compile_vulkan_shader(
        std::path::Path::new("resources/vulkan/matmul.comp"),
        "teamy_tts_matmul.comp.spv",
    );
    let embedding_output = compile_vulkan_shader(
        std::path::Path::new("resources/vulkan/embedding.comp"),
        "teamy_tts_embedding.comp.spv",
    );
    let conv1d_output = compile_vulkan_shader(
        std::path::Path::new("resources/vulkan/conv1d.comp"),
        "teamy_tts_conv1d.comp.spv",
    );
    let conv_transpose1d_output = compile_vulkan_shader(
        std::path::Path::new("resources/vulkan/conv_transpose1d.comp"),
        "teamy_tts_conv_transpose1d.comp.spv",
    );
    let elementwise_output = compile_vulkan_shader(
        std::path::Path::new("resources/vulkan/elementwise.comp"),
        "teamy_tts_elementwise.comp.spv",
    );
    let linear_output = compile_vulkan_shader(
        std::path::Path::new("resources/vulkan/linear.comp"),
        "teamy_tts_linear.comp.spv",
    );
    let length_regulate_output = compile_vulkan_shader(
        std::path::Path::new("resources/vulkan/length_regulate.comp"),
        "teamy_tts_length_regulate.comp.spv",
    );
    let lstm_output = compile_vulkan_shader(
        std::path::Path::new("resources/vulkan/lstm.comp"),
        "teamy_tts_lstm.comp.spv",
    );
    let batch_norm_output = compile_vulkan_shader(
        std::path::Path::new("resources/vulkan/batch_norm.comp"),
        "teamy_tts_batch_norm.comp.spv",
    );
    let max_pool1d_output = compile_vulkan_shader(
        std::path::Path::new("resources/vulkan/max_pool1d.comp"),
        "teamy_tts_max_pool1d.comp.spv",
    );
    let gru_output = compile_vulkan_shader(
        std::path::Path::new("resources/vulkan/gru.comp"),
        "teamy_tts_gru.comp.spv",
    );
    let copy_channels_output = compile_vulkan_shader(
        std::path::Path::new("resources/vulkan/copy_channels.comp"),
        "teamy_tts_copy_channels.comp.spv",
    );
    println!(
        "cargo:rustc-env=TEAMY_TTS_VECTOR_ADD_SPV={}",
        vector_add_output.display()
    );
    println!(
        "cargo:rustc-env=TEAMY_TTS_MATMUL_SPV={}",
        matrix_multiply_output.display()
    );
    println!(
        "cargo:rustc-env=TEAMY_TTS_EMBEDDING_SPV={}",
        embedding_output.display()
    );
    println!(
        "cargo:rustc-env=TEAMY_TTS_CONV1D_SPV={}",
        conv1d_output.display()
    );
    println!(
        "cargo:rustc-env=TEAMY_TTS_CONV_TRANSPOSE1D_SPV={}",
        conv_transpose1d_output.display()
    );
    println!(
        "cargo:rustc-env=TEAMY_TTS_ELEMENTWISE_SPV={}",
        elementwise_output.display()
    );
    println!(
        "cargo:rustc-env=TEAMY_TTS_LINEAR_SPV={}",
        linear_output.display()
    );
    println!(
        "cargo:rustc-env=TEAMY_TTS_LENGTH_REGULATE_SPV={}",
        length_regulate_output.display()
    );
    println!(
        "cargo:rustc-env=TEAMY_TTS_LSTM_SPV={}",
        lstm_output.display()
    );
    println!(
        "cargo:rustc-env=TEAMY_TTS_BATCH_NORM_SPV={}",
        batch_norm_output.display()
    );
    println!(
        "cargo:rustc-env=TEAMY_TTS_MAX_POOL1D_SPV={}",
        max_pool1d_output.display()
    );
    println!("cargo:rustc-env=TEAMY_TTS_GRU_SPV={}", gru_output.display());
    println!(
        "cargo:rustc-env=TEAMY_TTS_COPY_CHANNELS_SPV={}",
        copy_channels_output.display()
    );
}

/// `CubeCL`'s first Vulkan/SPIR-V kernel compilation can recurse deeply while
/// building the graph. The Windows default executable stack is too small for
/// that path, so give the CLI a larger main-thread stack when linking it.
fn add_windows_main_stack() {
    if cfg!(windows) {
        println!("cargo:rustc-link-arg-bin=teamy-tts=/STACK:67108864");
    }
}

fn compile_vulkan_shader(source: &std::path::Path, output_name: &str) -> std::path::PathBuf {
    println!("cargo:rerun-if-changed={}", source.display());
    let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR is set"))
        .join(output_name);
    let compiler = std::env::var_os("GLSLC")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("VULKAN_SDK").map(|sdk| {
                let mut path = std::path::PathBuf::from(sdk);
                path.push("Bin");
                path.push(if cfg!(windows) { "glslc.exe" } else { "glslc" });
                path
            })
        });
    let compiler = compiler.unwrap_or_else(|| std::path::PathBuf::from("glslc"));
    let status = std::process::Command::new(&compiler)
        .args(["-fshader-stage=compute", "-o"])
        .arg(&output)
        .arg(source)
        .status()
        .unwrap_or_else(|error| {
            panic!(
                "failed to run GLSLC at {}; set GLSLC or install the Vulkan SDK: {error}",
                compiler.display()
            )
        });
    assert!(
        status.success(),
        "GLSLC failed to compile {}",
        source.display()
    );
    output
}

/// Embeds Windows resources (like application icon) into the executable.
fn add_exe_resources() {
    println!("cargo:rerun-if-changed=resources");

    embed_resource::compile("resources/app.rc", embed_resource::NONE)
        .manifest_required()
        .expect("failed to embed resources");
}

/// In your code you can now access git metadata using
/// ```rust
/// let git_rev = option_env!("GIT_REVISION").unwrap_or("unknown");
/// ```
fn add_git_metadata() {
    add_git_metadata_inputs();

    // Try to get a short git revision; on failure, set to "unknown".
    let rev =
        git_output(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    let branch = git_output(&["branch", "--show-current"]).unwrap_or_else(|| {
        git_output(&["describe", "--tags", "--exact-match"])
            .unwrap_or_else(|| "detached".to_string())
    });
    let repository = git_output(&["remote", "get-url", "origin"])
        .or_else(|| std::env::var("CARGO_PKG_REPOSITORY").ok())
        .unwrap_or_else(|| "unknown".to_string());
    let dirty = match git_output_allow_empty(&["status", "--short", "--untracked-files=no"]) {
        Some(status) if status.is_empty() => "clean",
        Some(_) => "dirty",
        None => "unknown",
    };

    println!("cargo:rustc-env=GIT_REVISION={rev}");
    println!("cargo:rustc-env=GIT_BRANCH={branch}");
    println!("cargo:rustc-env=GIT_REPOSITORY_URL={repository}");
    println!("cargo:rustc-env=GIT_WORKTREE_STATUS={dirty}");
}

/// Re-run the build script when the current git metadata changes.
fn add_git_metadata_inputs() {
    if let Some(git_config_path) = git_output(&["rev-parse", "--git-path", "config"]) {
        println!("cargo:rerun-if-changed={git_config_path}");
    }

    if let Some(head_path) = git_output(&["rev-parse", "--git-path", "HEAD"]) {
        println!("cargo:rerun-if-changed={head_path}");
    }

    if let Some(index_path) = git_output(&["rev-parse", "--git-path", "index"]) {
        println!("cargo:rerun-if-changed={index_path}");
    }

    if let Some(head_ref) = git_output(&["symbolic-ref", "--quiet", "HEAD"])
        && let Some(head_ref_path) = git_output(&["rev-parse", "--git-path", &head_ref])
    {
        println!("cargo:rerun-if-changed={head_ref_path}");
    }
}

fn git_output(args: &[&str]) -> Option<String> {
    git_output_allow_empty(args).filter(|s| !s.is_empty())
}

fn git_output_allow_empty(args: &[&str]) -> Option<String> {
    let mut command = Command::new("git");
    if let Some(manifest_dir) = std::env::var_os("CARGO_MANIFEST_DIR") {
        command
            .arg("-c")
            .arg(format!("safe.directory={}", manifest_dir.to_string_lossy()));
    }

    command
        .args(args)
        .output()
        .ok()
        .and_then(|o| o.status.success().then_some(o.stdout))
        .and_then(|v| String::from_utf8(v).ok())
        .map(|s| s.trim().to_string())
}

/// Capture build time as a UTC instant so the runtime can render it in the user's local timezone.
fn add_build_timestamp() {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after Unix epoch")
        .as_secs();

    println!("cargo:rustc-env=BUILD_TIMESTAMP_UNIX={timestamp}");
}
