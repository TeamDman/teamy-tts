#include <torch/script.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {

struct Runtime {
    torch::jit::Module glados;
    torch::jit::Module vocoder;
    torch::Device vocoder_device;
};

void set_error(char **error, const std::string &message) {
    if (error == nullptr) {
        return;
    }
    *error = static_cast<char *>(std::malloc(message.size() + 1));
    if (*error != nullptr) {
        std::memcpy(*error, message.c_str(), message.size() + 1);
    }
}

torch::Tensor find_mel_post(const torch::jit::IValue &output) {
    const auto dictionary = output.toGenericDict();
    for (const auto &entry : dictionary) {
        if (entry.key().isString() && entry.key().toStringRef() == "mel_post") {
            return entry.value().toTensor();
        }
    }
    throw std::runtime_error("GLaDOS output did not contain mel_post");
}

} // namespace

extern "C" {

void *teamy_tts_torchscript_create(
    const char *glados_path,
    const char *vocoder_path,
    int device_index,
    char **error) {
    try {
        auto *runtime = new Runtime{
            torch::jit::load(glados_path, torch::Device(torch::kCPU)),
            torch::jit::load(vocoder_path, torch::Device(torch::kCUDA, device_index)),
            torch::Device(torch::kCUDA, device_index),
        };
        runtime->glados.eval();
        runtime->vocoder.eval();
        return runtime;
    } catch (const std::exception &exception) {
        set_error(error, exception.what());
        return nullptr;
    }
}

int teamy_tts_torchscript_synthesize(
    void *opaque_runtime,
    const int64_t *token_values,
    std::size_t token_count,
    const float *speaker_values,
    std::size_t speaker_count,
    float alpha,
    float **audio_values,
    std::size_t *audio_count,
    char **error) {
    try {
        auto &runtime = *static_cast<Runtime *>(opaque_runtime);
        torch::NoGradGuard no_grad;
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        auto speaker = torch::from_blob(
                           const_cast<float *>(speaker_values),
                           {1, static_cast<long>(speaker_count)},
                           options)
                           .clone();
        auto token_options = torch::TensorOptions().dtype(torch::kInt64);
        auto tokens = torch::from_blob(
                          const_cast<int64_t *>(token_values),
                          {1, static_cast<long>(token_count)},
                          token_options)
                          .clone();
        auto output = runtime.glados.run_method(
            "generate_jit", tokens, speaker, static_cast<double>(alpha));
        auto mel = find_mel_post(output);
        auto audio = runtime.vocoder.forward({mel.to(runtime.vocoder_device)})
                         .toTensor()
                         .to(torch::kCPU)
                         .contiguous();
        const auto flattened = audio.flatten().to(torch::kFloat32);
        const auto count = static_cast<std::size_t>(flattened.numel());
        auto *result = static_cast<float *>(std::malloc(count * sizeof(float)));
        if (result == nullptr) {
            throw std::bad_alloc();
        }
        std::memcpy(result, flattened.data_ptr<float>(), count * sizeof(float));
        *audio_values = result;
        *audio_count = count;
        return 0;
    } catch (const std::exception &exception) {
        set_error(error, exception.what());
        return 1;
    }
}

void teamy_tts_torchscript_free_audio(float *audio_values) {
    std::free(audio_values);
}

void teamy_tts_torchscript_destroy(void *opaque_runtime) {
    delete static_cast<Runtime *>(opaque_runtime);
}

void teamy_tts_torchscript_free_error(char *error) {
    std::free(error);
}

} // extern "C"
