"""Runtime contract schema path resolution for vitals telemetry.

At runtime the v2 vitals JSON schema is loaded from a canonical on-device
path so that the Yocto image is the single source of truth for the contract.

Resolution order:
1. ``MEDTECH_VITALS_SCHEMA`` environment variable (if set and non-empty).
2. Default Yocto rootfs path: ``/usr/share/medtech/contracts/vitals/current.json``.

If the resolved file is missing or unreadable the service must **hard-fail**
(raise ``FileNotFoundError``) so that systemd / the process supervisor can
report the failure and stop the unit — a system running without a contract
file is not in a defined state.

For tests and CI the vendored copy at ``contracts/vitals/v2.0.json`` can be
injected via the environment variable without changing any code paths:

    MEDTECH_VITALS_SCHEMA=contracts/vitals/v2.0.json pytest ...
"""

import os
import pathlib

# Default canonical location on the Yocto rootfs.
_DEFAULT_SCHEMA_PATH = "/usr/share/medtech/contracts/vitals/current.json"

# Environment variable that overrides the default path.
_ENV_VAR = "MEDTECH_VITALS_SCHEMA"


def resolve_schema_path() -> pathlib.Path:
    """Return the resolved, validated path to the vitals JSON schema file.

    Checks the ``MEDTECH_VITALS_SCHEMA`` environment variable first; falls
    back to the default Yocto rootfs location.

    Returns:
        A :class:`pathlib.Path` object pointing to the schema file.

    Raises:
        FileNotFoundError: If the resolved path does not exist or is not
            readable.  The caller must treat this as a fatal startup error.
    """
    raw = os.environ.get(_ENV_VAR, "").strip()
    schema_path = pathlib.Path(raw) if raw else pathlib.Path(_DEFAULT_SCHEMA_PATH)

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Vitals contract schema not found at '{schema_path}'. "
            f"Install the medtech-telemetry-contract package or set the "
            f"{_ENV_VAR} environment variable to the schema file path."
        )

    if not os.access(schema_path, os.R_OK):
        raise FileNotFoundError(
            f"Vitals contract schema at '{schema_path}' is not readable. "
            f"Check file permissions or set {_ENV_VAR} to an accessible path."
        )

    return schema_path
