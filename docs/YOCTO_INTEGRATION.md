# Yocto Integration: Sepsis Model Delivery

This guide explains how Yocto consumes the model produced by CI and how to pin
model versions for deterministic releases.

## Delivery Contract

- Canonical model path in repository: `models/imx8-compatible-sepsis.tflite`
- Model updates are auto-committed by CI after successful retraining
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

## Security and Credentials

- CI uses built-in `GITHUB_TOKEN` for model commit/tag actions.
- No personal access token is required for normal model delivery flow.
