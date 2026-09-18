# Teamy TTS desktop SAPI voice

Supported target: Windows x64, native CUDA build, a compatible NVIDIA GPU,
and the installed GLaDOS FP32 weights. The COM DLL has no CUDA or LibTorch
dependency. It sends text over a local named pipe to the native worker and
returns 22,050 Hz mono PCM16 to SAPI. SAPI owns playback.

## Install and use

1. Run the repository's `update.ps1` to install the CLI, vendor DLLs, weights
   and a versioned SAPI adapter.
2. In an administrator terminal, run `teamy-tts sapi install`. It copies the
   adapter into a versioned directory under Program Files and registers the
   64-bit COM class and SAPI voice. It does not select a default voice.
3. In a normal terminal, run `teamy-tts sapi status`, then
   `teamy-tts sapi test --text "Hello, friend" --output .\sapi-test.wav`.
   Omit `--output` to hear the test through SAPI.
4. To use applications that rely on the default voice, select
   **Teamy GLaDOS (native)** in Windows' legacy Speech properties. Restart
   those applications after changing the default or updating the adapter.

`sapi status` reports registration, enumeration, the unchanged system default,
configured worker/model paths, and worker state without starting the worker.
`sapi stop` waits for the worker to exit. `sapi uninstall` removes the machine
voice and COM registration; run it as administrator after selecting another
default voice. Versioned adapter files are retained so already-running hosts
can finish safely. They can be removed after closing those applications.

`--per-user` is available on install/status/test/uninstall for development.
Such a voice works with explicit `SetVoice` clients but is absent from normal
Windows SAPI enumeration; it is not the Minecraft setup.

## Worker and cancellation

The adapter connects to `serve`, starting it hidden only when no worker is
present. Concurrent starters converge on one named-pipe instance. CUDA and the
model stay on one owning inference thread. No Windows service, login task,
firewall rule, network listener or Python runtime is required.

The default worker exits after 120 seconds without clients. For manual use:

```powershell
teamy-tts serve --idle-seconds 600
```

The pipe is scoped to the current Windows user and session, denies remote
clients, and checks the server's user identity. An elevated SAPI host may
connect to an existing worker, but cannot automatically launch a user-installed
worker with its elevated token. Start `serve` in a normal terminal first if
speech is needed in an administrator application.

SAPI's ordinary asynchronous queue preserves chat order. Purge replaces prior
speech; `Skip("Sentence", INT_MAX)` clears the current narration. The adapter
polls SAPI actions during startup, inference and PCM writes. Cancellation is
per connection; stale results are discarded. A running GPU operation finishes
its current bounded text chunk before the next job can use the GPU.

Limits are explicit: 16 clients, eight queued inference jobs, one outstanding
job per connection, 280 characters per inference chunk, 16,384 UTF-16 units
per SAPI utterance, 30-second startup and 60-second synthesis deadlines. A
busy or failed worker returns an error instead of accumulating unbounded work.
Long sentences are split for inference without treating each chunk as a new
sentence for SAPI Skip. Sentence detection uses punctuation, common English
abbreviations and line breaks; it is not a full linguistic parser.

Volume is applied to output PCM. SAPI rate -10 through +10 maps to the native
duration control. Full SAPI XML/prosody markup, word-level events, WinRT voices,
32-bit applications and non-NVIDIA inference are outside this integration.

## Minecraft acceptance

Minecraft Java 1.19.2's resolved Mojang `text2speech:1.16.7` uses desktop
`ISpVoice`: asynchronous plain-text Speak flags 17, replacement flags 19,
and `Skip("Sentence", INT_MAX)`. It does not explicitly select a voice.

The real SAPI harness tests those calls, PCM equality with native inference,
FIFO ordering, cold-start purge, sentence skip, volume, rate, COM lifetime,
and preservation of the default voice. The worker probe tests concurrent
startup, client isolation, stale output, bounded requests, restart, idle exit
and missing-model errors. Capture files avoid changing the user's audio output.

Actual Minecraft gameplay has not been validated. After selecting the voice:

1. Restart Minecraft 1.19.2 and enable its narrator.
2. Move rapidly between menu controls. Obsolete labels should stop promptly.
3. Send two chat messages. Both should play in order.
4. Clear/disable narration during a long message, then re-enable it.
5. Leave narration idle for two minutes, then speak again to check cold restart.

## Developer checks

All runtime checks use release builds. The `test-fixture` feature is only for
the test adapter and must not be enabled in an installed production DLL.

```powershell
cargo build --release --manifest-path ipc/Cargo.toml --locked
cargo build --release --manifest-path sapi/Cargo.toml --features test-fixture --locked
$env:TEAMY_SAPI_ALLOW_TEST_REGISTRATION = '1'
ipc/target/release/worker_probe.exe <worker-exe> <receipt-directory>
sapi/target/release/sapi-probe.exe <test-dll> <fixture-receipt-directory>
sapi/target/release/sapi-probe.exe <test-dll> <native-receipt-directory> <worker-exe>
cargo build --release --manifest-path sapi/Cargo.toml --locked
python sapi/tools/check-dll.py <production-dll>
```

Test registration uses a separate per-user CLSID/token and is removed by the
harness. The production installer builds without the fixture feature.
