locals {
  models_archive_sha256 = filesha256(local.models_archive_source)
}
