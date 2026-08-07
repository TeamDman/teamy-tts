# R2 exposes an S3-compatible API. The AWS provider reads
# AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY from the environment.
provider "aws" {
  alias  = "r2"
  region = "us-east-1"

  skip_credentials_validation = true
  skip_region_validation      = true
  skip_requesting_account_id  = true
  skip_metadata_api_check     = true

  endpoints {
    s3 = "https://${var.cloudflare_account_id}.r2.cloudflarestorage.com"
  }
}
