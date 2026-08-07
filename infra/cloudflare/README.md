# Teamy TTS Cloudflare infrastructure

This configuration creates the `teamy-tts-models` R2 bucket, its
incomplete-multipart-upload lifecycle rule, and publishes the repository's
`models.zip` as an immutable, SHA-256-addressed object.

The configuration does not create a public bucket or custom domain. Public
download exposure is a separate decision for the model source contract.

Terraform state uses a private Azure Storage backend configured through the
ignored local `backend.hcl` file, with the recommended key
`teamy-tts/cloudflare.tfstate`. Local state, plans, provider working
directories, and credentials are ignored by the repository.

## Credentials

From this directory, dot-source both credential helpers:

```powershell
. ..\..\get-cloudflare-token.ps1
. .\get-r2-s3-credentials.ps1
```

The Cloudflare helper supplies `CLOUDFLARE_API_TOKEN`. The R2 helper supplies
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` for the S3-compatible R2 API.
The referenced 1Password item must have `access key id` and
`secret access key` fields in addition to its existing `credential` field.

## Initialize and plan

The bucket name, bucket location, archive path, and object prefix are explicit
literals in the Terraform files. The Cloudflare account ID is supplied through
`TF_VAR_cloudflare_account_id` so it is not committed to this public
repository.

The bucket uses Cloudflare's `ENAM` (Eastern North America) location hint. R2
does not currently expose a Canada-specific hint; Cloudflare maps Canada
Central/Toronto to `ENAM`, making it the closest supported hint for Ottawa.
This is a best-effort placement hint, not a Canadian data-residency guarantee.

```powershell
Copy-Item .\backend.hcl.example .\backend.hcl
$env:TF_VAR_cloudflare_account_id = "<cloudflare-account-id>"
terraform init -backend-config=backend.hcl
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

The archive object key contains its SHA-256. An independent HEAD/download
verification should still be recorded as the publication receipt.
