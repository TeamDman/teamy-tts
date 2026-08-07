# teamy-tts

Local Rust/Burn text-to-speech CLI based on the GLaDOS TTS pipeline.

Target command surface:

~~~powershell
teamy-tts say --model glados "hello!" --output output.wav
teamy-tts model list
teamy-tts model acquire-unprepared Teamy
teamy-tts model acquire-unprepared R2D2FISH-OneDrive
teamy-tts model prepare glados
~~~

The executable should be downloadable and usable without a checkout of the
upstream Python repository. Model assets are prepared separately, verified,
cached, and reported by the CLI.

This project is licensed under the [Mozilla Public License 2.0](LICENSE).

The raw upstream archive is a separate acquisition step. The Teamy source
will be rehosted in Cloudflare R2, while R2D2FISH-OneDrive represents the
upstream-maintainer source. Terraform owns the R2 infrastructure; publication
and post-upload verification must prove that the immutable archive exists
before the CLI catalog points at it.

This repository is currently in planning. The executable plan is in
[PLAN.md](PLAN.md).

Cloudflare R2 infrastructure is defined in
[infra/cloudflare](infra/cloudflare/README.md).

## Terraform Cloudflare session

The public repository does not contain owner-specific 1Password references or
cloud account identifiers. Copy the example helpers, set your own references,
and load the credentials into the current PowerShell session:

```powershell
Copy-Item .\get-cloudflare-token.ps1.example .\get-cloudflare-token.ps1
Copy-Item .\infra\cloudflare\get-r2-s3-credentials.ps1.example .\infra\cloudflare\get-r2-s3-credentials.ps1
$env:TEAMY_TTS_CLOUDFLARE_OP_REF = "op://<vault>/<item>/credential"
$env:TEAMY_TTS_CLOUDFLARE_OP_ITEM = "op://<vault>/<item>"
Set-Location .\infra\cloudflare
. ..\..\get-cloudflare-token.ps1
. .\get-r2-s3-credentials.ps1
```

Create a local ignored `backend.hcl` from `infra/cloudflare/backend.hcl.example`
with your Azure state storage details, set the Cloudflare account ID, and
initialize Terraform with that file:

```powershell
Copy-Item .\backend.hcl.example .\backend.hcl
$env:TF_VAR_cloudflare_account_id = "<cloudflare-account-id>"
terraform init -backend-config=backend.hcl
terraform plan
```

The helpers use `op read` and do not write secrets to repository files. They
must be dot-sourced; running them as separate processes would not preserve
their environment variables for Terraform. Clear credentials when finished
if this terminal will remain open:

```powershell
Remove-Item Env:\CLOUDFLARE_API_TOKEN -ErrorAction SilentlyContinue
Remove-Item Env:\AWS_ACCESS_KEY_ID -ErrorAction SilentlyContinue
Remove-Item Env:\AWS_SECRET_ACCESS_KEY -ErrorAction SilentlyContinue
Remove-Item Env:\TF_VAR_cloudflare_account_id -ErrorAction SilentlyContinue
```
