// Windows MSVC drops the torch_cuda.dll import when no object references a
// CUDA symbol. That leaves LibTorch's global CUDA context disabled even though
// the CUDA runtime and driver are installed. This one-symbol anchor keeps the
// import alive; it is a loader workaround, not a second tensor API.
namespace at::cuda {
void CachingHostAllocator_emptyCache();
}

extern "C" void teamy_tts_force_torch_cuda() {
    at::cuda::CachingHostAllocator_emptyCache();
}
