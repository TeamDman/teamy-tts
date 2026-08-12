use sha2::Digest;
use sha2::Sha256;
use std::process::Command;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=Cargo.lock");
    println!("cargo:rerun-if-changed=src");
    add_source_fingerprint();
    add_exe_resources();
    add_windows_cuda_link_anchor();
    add_git_metadata();
    add_build_timestamp();
    if cfg!(windows) {
        println!("cargo:rustc-link-arg-bin=teamy-tts=/STACK:67108864");
    }
}

fn add_windows_cuda_link_anchor() {
    if !cfg!(windows) {
        return;
    }
    let Some(libtorch) = std::env::var_os("LIBTORCH") else {
        return;
    };
    let libtorch = std::path::PathBuf::from(libtorch);
    let cuda_import_library = libtorch.join("lib").join("torch_cuda.lib");
    if !cuda_import_library.is_file() {
        return;
    }

    println!("cargo:rustc-check-cfg=cfg(teamy_tts_cuda_link)");
    println!("cargo:rustc-cfg=teamy_tts_cuda_link");
    println!("cargo:rerun-if-changed=src/native_glados/cuda_link_anchor.cpp");
    println!("cargo:rerun-if-env-changed=LIBTORCH");

    cc::Build::new()
        .cpp(true)
        .file("src/native_glados/cuda_link_anchor.cpp")
        .include(libtorch.join("include"))
        .include(libtorch.join("include/torch/csrc/api/include"))
        .flag_if_supported("/std:c++17")
        .compile("teamy_tts_cuda_link_anchor");
}

fn add_source_fingerprint() {
    let mut files = Vec::new();
    for root in [
        std::path::Path::new("src"),
        std::path::Path::new("build.rs"),
        std::path::Path::new("Cargo.toml"),
        std::path::Path::new("Cargo.lock"),
    ] {
        collect_files(root, &mut files);
    }
    files.sort();
    let mut digest = Sha256::new();
    for path in files {
        digest.update(path.to_string_lossy().replace('\\', "/").as_bytes());
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
    } else if path.is_dir() {
        for entry in std::fs::read_dir(path)
            .unwrap_or_else(|error| panic!("failed to enumerate {}: {error}", path.display()))
        {
            collect_files(
                &entry
                    .unwrap_or_else(|error| {
                        panic!("failed to enumerate {}: {error}", path.display())
                    })
                    .path(),
                files,
            );
        }
    }
}

fn add_exe_resources() {
    println!("cargo:rerun-if-changed=resources");
    embed_resource::compile("resources/app.rc", embed_resource::NONE)
        .manifest_required()
        .expect("failed to embed resources");
}

fn add_git_metadata() {
    let rev =
        git_output(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    let branch = git_output(&["branch", "--show-current"])
        .or_else(|| git_output(&["describe", "--tags", "--exact-match"]))
        .unwrap_or_else(|| "detached".to_string());
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

fn git_output(args: &[&str]) -> Option<String> {
    git_output_allow_empty(args).filter(|value| !value.is_empty())
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
        .and_then(|output| output.status.success().then_some(output.stdout))
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .map(|value| value.trim().to_string())
}

fn add_build_timestamp() {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock should be after Unix epoch")
        .as_secs();
    println!("cargo:rustc-env=BUILD_TIMESTAMP_UNIX={timestamp}");
}
