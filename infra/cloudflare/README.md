# Teamy TTS Cloudflare infrastructure

This configuration creates the `teamy-tts-models` R2 bucket, enables its
Cloudflare-managed public development URL, and publishes the repository's
`models.zip` and the generated native bundle ZIP as immutable,
SHA-256-addressed objects.

Terraform exposes the two concrete public object URLs as outputs. Those
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
```

The object keys contain their SHA-256. Build the native bundle first with
`tools/package-native-bundle.ps1`; Terraform expects it at
`artifacts/teamy-tts-glados-new-native-bundle.zip`. An independent
HEAD/download verification should still be recorded as the publication
receipt. The managed `r2.dev` endpoint is intended for development and is
rate-limited; use a Cloudflare custom domain before treating this as a
production distribution endpoint.
