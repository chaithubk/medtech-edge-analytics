# Yocto Integration: Sepsis Model Delivery

This guide explains how Yocto consumes the model produced by CI and how to pin
model versions for deterministic releases.

## Delivery Contract

- Canonical model path in repository: `models/imx8-compatible-sepsis.tflite`
- Model updates are produced by CI after successful retraining.
- NOTE: For safety and branch protection compliance CI creates or updates a
	reviewable pull request (`model-update-sepsis-model`) containing the updated
	model artifact; maintainers should review and merge the PR to incorporate
	the model into `main`.
- CI also creates model tags in format: `models/<synthea-version>-<timestamp>`
- No runtime download is required on target devices

## Integration Pattern

Use this repository as a source in your Yocto recipe, install the model during
`do_install`, and pin `SRCREV` to a validated commit SHA.

### Example BitBake Snippet

```bitbake
SUMMARY = "MedTech sepsis model artifact"
LICENSE = "CLOSED"

SRC_URI = "git://github.com/chaithubk/medtech-edge-analytics.git;protocol=https;branch=main"

# Pin to a tested commit that contains the desired model tag.
SRCREV = "<commit-sha>"

S = "${WORKDIR}/git"

do_install() {
	install -d ${D}${datadir}/medtech/models
	install -m 0644 ${S}/models/imx8-compatible-sepsis.tflite \
		${D}${datadir}/medtech/models/imx8-compatible-sepsis.tflite
}

FILES:${PN} += "${datadir}/medtech/models/imx8-compatible-sepsis.tflite"
```

## Update Procedure for New Model

1. Identify latest validated model tag from CI.
2. Resolve tag to commit SHA.
3. Update `SRCREV` in the Yocto recipe.
4. Rebuild image and run smoke validation on target.

## Runtime Service Configuration

Point your inference service to the installed model path, for example:

- `MODEL_PATH=/usr/share/medtech/models/imx8-compatible-sepsis.tflite`

This keeps deployment deterministic and aligned with Yocto image contents.

## Local development parity

To make your development environment behave like the Yocto image, the
devcontainer is configured to copy the vendored schema and canonical model into
the expected rootfs locations on container setup. This mirrors production
runtime paths and avoids needing to set environment variables in local runs.

If you prefer not to copy files into `/usr/share/medtech/`, you can instead
set the environment variables locally:

```bash
export MEDTECH_VITALS_SCHEMA=contracts/vitals/vitals.schema.json
export MODEL_PATH=models/imx8-compatible-sepsis.tflite
```

Both approaches are supported; the devcontainer copy provides the closest
parity with the Yocto image and is recommended for day-to-day development.

## Security and Credentials

- CI uses built-in `GITHUB_TOKEN` for model commit/tag actions.
- No personal access token is required for normal model delivery flow.
