use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=kernels/ops.cu");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=GLADOS_CUDA_ARCH");
    let cuda = PathBuf::from(env::var_os("CUDA_PATH").expect("CUDA_PATH must name a CUDA toolkit"));
    let arch = env::var("GLADOS_CUDA_ARCH").unwrap_or_else(|_| "89".into());
    assert!(
        arch.chars().all(|c| c.is_ascii_alphanumeric()),
        "invalid CUDA architecture"
    );
    cc::Build::new()
        .cuda(true)
        .cudart("shared")
        .flag("-std=c++17")
        .flag("-O3")
        .flag("-lineinfo")
        .flag(format!("-gencode=arch=compute_{arch},code=sm_{arch}"))
        .include(cuda.join("include"))
        .file("kernels/ops.cu")
        .compile("glados_cuda");
    let lib = if cfg!(windows) {
        cuda.join("lib/x64")
    } else {
        cuda.join("lib64")
    };
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=cublas");
}
