locals {
  native_bundle_archive_sha256 = filesha256(local.native_bundle_archive_source)
}
