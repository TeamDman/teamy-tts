# Teamy TTS Cloudflare infrastructure

This configuration creates the `teamy-tts-models` R2 bucket, enables its
Cloudflare-managed public development URL, and publishes the repository's
`models.zip`, the legacy TorchScript bundle, and the CUDA safetensors ZIP as immutable,
SHA-256-addressed objects.

Terraform exposes the three concrete public object URLs as outputs. Those
outputs are intentionally used to bake the Teamy source defaults into the Rust
application after the managed domain has been created.

The Cloudflare provider manages the R2 bucket, lifecycle rule, and public
domain. The AWS provider is present only because R2 exposes an AWS S3-compatible
object API; its `aws_s3_object` resources upload into Cloudflare R2 and do not
create or use AWS storage.

Terraform state uses the private Azure Storage backend literals in
`terraform.tf`, with the key `teamy-tts/cloudflare.tfstate`. Local state,
plans, provider working directories, and credentials are ignored by the
repository.

## Credentials

From this directory, dot-source the repository credential loader:

```powershell
. ..\..\get-cloudflare-token.ps1
```

The loader supplies `CLOUDFLARE_API_TOKEN`, `AWS_ACCESS_KEY_ID`, and
`AWS_SECRET_ACCESS_KEY` from the fixed 1Password item. The AWS-named values
are Cloudflare R2 access keys, not AWS account credentials.

## Initialize and plan

The account ID, bucket name, bucket location, archive path, object prefix, and
state backend are explicit literals in the Terraform files. The account ID
and backend identifiers are not secrets; the token and R2 access keys remain
behind `op`.

The bucket uses Cloudflare's `ENAM` (Eastern North America) location hint. R2
does not currently expose a Canada-specific hint; Cloudflare maps Canada
Central/Toronto to `ENAM`, making it the closest supported hint for Ottawa.
This is a best-effort placement hint, not a Canadian data-residency guarantee.

```powershell
terraform init
terraform fmt -check
terraform validate
terraform plan
```

For a fresh checkout, `terraform init` configures the remote state backend. An
authorized operator must load the Cloudflare and R2 credentials before
planning or applying.

Applying also uploads the large archive:

```powershell
terraform apply
```

After applying, capture the URLs that the application will use:

```powershell
terraform output -raw teamy_raw_source_url
terraform output -raw teamy_native_source_url
terraform output -raw teamy_cuda_source_url
```

Each upload's source and hash use the same local path. The legacy `.pt` ZIP
defaults to `artifacts/teamy-tts-glados-new-tch-native-bundle.zip`; the CUDA ZIP
defaults to `artifacts/teamy-tts-glados-native-v1.zip`. Archive source variables
allow another local path without changing which file supplies the hash.

Build the CUDA package with `python tools/package-cuda-bundle.py --help` and
the known validated export. It contains only FP32 safetensors, frontend data
and a runtime manifest. Its checked-in catalog pins the exact size, archive
hash and individual file hashes; Terraform rejects a mismatched upload.

The historical `native/glados/ab663a68...` object contains the original `.bpk`
bundle. A Terraform `moved` block transfers its state to
`native_bundle_historical`, preserving its key and bytes. The corrected
TorchScript bundle and CUDA bundle use separate new resources. Existing
objects have `prevent_destroy`; review a new release plan before changing a
pinned resource. Do not replace an old key with different model bytes.

Inspect a saved plan before applying. This migration should create two model
objects, move the historical state address, and delete or replace nothing.
Record an independent anonymous full download and SHA-256 check, plus access
checks for existing URLs. The first version uses the existing managed `r2.dev`
endpoint. A custom domain remains an optional later infrastructure change.
