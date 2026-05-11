"""Contract compatibility utilities for the telemetry schema.

Provides helpers for:
- Parsing SemVer version strings.
- Checking whether an incoming payload version is compatible with the pinned
  contract (same MAJOR, any MINOR/PATCH).
- Loading the vendored contract manifest (YAML).
- Assessing upgrade safety when bumping the pinned contract tag.

Usage example (upgrade safety check)::

    from src.utils.contract_compat import check_upgrade_safety

    result = check_upgrade_safety(from_version="2.1.0")
    if not result["safe"]:
        raise RuntimeError(f"BREAKING contract upgrade: {result['actions_required']}")

Runtime version compatibility check::

    from src.utils.contract_compat import is_compatible_version

    if not is_compatible_version(payload.get("version", "")):
        raise ValueError("Incompatible payload version")
"""

import json
import os
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple, cast

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_CONTRACTS_DIR_ENV_VAR = "MEDTECH_CONTRACTS_DIR"
_DEFAULT_CONTRACTS_DIR = pathlib.Path("/usr/share/medtech/contracts")
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _iter_contracts_dirs() -> List[pathlib.Path]:
    """Return candidate contract directories ordered by runtime preference."""
    candidates: List[pathlib.Path] = []

    env_contracts_dir = os.environ.get(_CONTRACTS_DIR_ENV_VAR, "").strip()
    if env_contracts_dir:
        candidates.append(pathlib.Path(env_contracts_dir))

    # Canonical Yocto/rootfs location used by containers and edge devices.
    candidates.append(_DEFAULT_CONTRACTS_DIR)

    # Local dev and test layouts (walk parent dirs of this module).
    module_path = pathlib.Path(__file__).resolve()
    for parent in module_path.parents:
        candidates.append(parent / "contracts")

    # Last resort: current working directory.
    candidates.append(pathlib.Path.cwd() / "contracts")

    unique: List[pathlib.Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _resolve_contract_file(relative_path: pathlib.Path) -> pathlib.Path:
    """Resolve a contract file path from known runtime/development locations."""
    dirs = _iter_contracts_dirs()
    for contracts_dir in dirs:
        candidate = contracts_dir / relative_path
        if candidate.exists():
            return candidate
    return dirs[0] / relative_path


_DEFAULT_MANIFEST_PATH = _resolve_contract_file(
    pathlib.Path("vitals") / "vitals.schema-manifest.yml"
)

# Pinned contract metadata file shared with platform orchestration.
_CONTRACT_PIN_PATH = _resolve_contract_file(pathlib.Path("vitals") / "contract-pin.json")
_VERSION_PIN_PATH = _resolve_contract_file(pathlib.Path("VITALS_CONTRACT_VERSION.txt"))


def _load_pinned_contract_version() -> str:
    """Resolve pinned contract version from metadata with safe fallbacks."""
    if _CONTRACT_PIN_PATH.exists():
        try:
            data = json.loads(_CONTRACT_PIN_PATH.read_text(encoding="utf-8"))
            tag = str(data.get("tag", "")).strip()
            if tag.startswith("v"):
                return tag[1:]
            if tag:
                return tag
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to parse %s; falling back to manifest.", _CONTRACT_PIN_PATH)

    if _VERSION_PIN_PATH.exists():
        try:
            version = _VERSION_PIN_PATH.read_text(encoding="utf-8").strip()
            if version.startswith("v"):
                return version[1:]
            if version:
                return version
        except OSError:
            logger.warning("Failed to read %s; falling back to manifest.", _VERSION_PIN_PATH)

    if _DEFAULT_MANIFEST_PATH.exists():
        try:
            import yaml  # type: ignore[import]  # noqa: PLC0415

            manifest = yaml.safe_load(_DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))
            if isinstance(manifest, dict):
                schema_version = str(
                    manifest.get("schema_version") or manifest.get("current_version") or ""
                ).strip()
                if schema_version:
                    return schema_version
        except Exception:
            logger.warning("Failed to resolve pinned contract version from manifest.")

    # Safe fallback keeps service behavior deterministic if metadata is missing.
    return "0.0.0"


# Pinned contract version for this consumer revision.
PINNED_CONTRACT_VERSION = _load_pinned_contract_version()

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")

# Legacy two-part version "2.0" accepted for backward compat during migration.
_LEGACY_2PART_RE = re.compile(r"^(\d+)\.(\d+)$")

# Environment variable to override the manifest path (mirrors the schema loader pattern).
_MANIFEST_ENV_VAR = "MEDTECH_VITALS_MANIFEST"


# ---------------------------------------------------------------------------
# SemVer helpers
# ---------------------------------------------------------------------------


def parse_semver(version: str) -> Tuple[int, int, int]:
    """Parse a SemVer string into ``(major, minor, patch)`` integers.

    Args:
        version: Version string in ``MAJOR.MINOR.PATCH`` format.

    Returns:
        Tuple ``(major, minor, patch)``.

    Raises:
        ValueError: If the string is not valid ``MAJOR.MINOR.PATCH`` SemVer.
    """
    m = _SEMVER_RE.match(version)
    if not m:
        raise ValueError(
            f"Invalid SemVer format: '{version}'. Expected MAJOR.MINOR.PATCH (e.g. '2.0.0')."
        )
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def is_compatible_version(
    payload_version: object,
    pinned_version: str = PINNED_CONTRACT_VERSION,
) -> bool:
    """Return ``True`` if ``payload_version`` is compatible with ``pinned_version``.

    Compatibility rules:

    - **Exact match** → always compatible.
    - **Same MAJOR, any MINOR/PATCH** → compatible (backward-compatible additions
      are safe; consumer ignores unknown fields at runtime).
    - **Different MAJOR** → incompatible (BREAKING change).
    - **Legacy two-part format** (e.g. ``"2.0"``) → accepted as ``"2.0.0"``
      during the migration window; emits a deprecation warning.

    Args:
        payload_version: Version string from the incoming payload.
        pinned_version:  Pinned contract version to compare against
                         (defaults to :data:`PINNED_CONTRACT_VERSION`).

    Returns:
        ``True`` if the payload can be safely processed.
    """
    if payload_version == pinned_version:
        return True

    # Reject missing or non-string versions explicitly.  This keeps caller
    # behaviour predictable (False/incompatible) instead of raising TypeError.
    if not isinstance(payload_version, str):
        logger.warning(
            "Payload version has invalid type %s (value=%r) — treating as incompatible.",
            type(payload_version).__name__,
            payload_version,
        )
        return False

    # Normalise legacy two-part format "M.N" → "M.N.0"
    legacy_m = _LEGACY_2PART_RE.match(payload_version or "")
    normalised = payload_version
    if legacy_m:
        normalised = f"{legacy_m.group(1)}.{legacy_m.group(2)}.0"
        logger.warning(
            "Payload uses legacy two-part version '%s'. "
            "Normalised to '%s' for compatibility check. "
            "Producers should emit full SemVer (e.g. '2.0.0').",
            payload_version,
            normalised,
        )

    try:
        p_major, _p_minor, _p_patch = parse_semver(normalised)
        pin_major, _pin_minor, _pin_patch = parse_semver(pinned_version)
    except ValueError:
        logger.warning(
            "Cannot parse version string '%s' — treating as incompatible.",
            payload_version,
        )
        return False

    if p_major != pin_major:
        return False  # Different MAJOR → BREAKING

    # Same MAJOR: any MINOR / PATCH is backward-compatible for the consumer.
    return True


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: Optional[pathlib.Path] = None) -> Dict[str, Any]:
    """Load the vendored contract manifest YAML and return it as a dict.

    Resolution order for the manifest file:

    1. ``manifest_path`` argument (if supplied).
    2. ``MEDTECH_VITALS_MANIFEST`` environment variable.
    3. Default: ``contracts/vitals/vitals.schema-manifest.yml`` in the repo root.

    Args:
        manifest_path: Optional explicit path to the manifest YAML file.

    Returns:
        Parsed manifest as a plain ``dict``.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
        ImportError: If PyYAML (``pyyaml``) is not installed.
    """
    if manifest_path is None:
        env_raw = os.environ.get(_MANIFEST_ENV_VAR, "").strip()
        manifest_path = pathlib.Path(env_raw) if env_raw else _DEFAULT_MANIFEST_PATH

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Contract manifest not found at '{manifest_path}'. "
            f"Run the 'Vendor Telemetry Contract' workflow or set {_MANIFEST_ENV_VAR}."
        )

    try:
        import yaml  # type: ignore[import]  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load the contract manifest. "
            "Install it with: pip install pyyaml"
        ) from exc

    with manifest_path.open(encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)

    if not isinstance(loaded, dict):
        raise ValueError(f"Contract manifest at '{manifest_path}' must be a mapping/object.")

    manifest: Dict[str, Any] = cast(Dict[str, Any], loaded)

    # Normalize upstream manifest variants to stable local keys expected by
    # consumer code and tests.
    if "schema_version" not in manifest and "current_version" in manifest:
        manifest["schema_version"] = manifest["current_version"]
    if "contract_repo" not in manifest:
        manifest["contract_repo"] = "chaithubk/medtech-telemetry-contract"
    if "canonical_schema_path" not in manifest:
        schema_file = manifest.get("schema_file", "vitals.schema.json")
        manifest["canonical_schema_path"] = f"schemas/vitals/{schema_file}"
    if "vendored_at" not in manifest and "release_date" in manifest:
        manifest["vendored_at"] = manifest["release_date"]

    return manifest


# ---------------------------------------------------------------------------
# Upgrade safety assessment
# ---------------------------------------------------------------------------


def check_upgrade_safety(
    from_version: str,
    manifest_path: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    """Assess whether upgrading from ``from_version`` to the pinned contract is safe.

    Reads the vendored manifest to determine the compatibility class and any
    required human actions before the upgrade can be merged.

    Args:
        from_version:  Currently deployed contract version (e.g. ``"2.0.0"``).
        manifest_path: Optional override for the manifest file path.

    Returns:
        A dict with keys:

        - ``safe`` (bool): ``True`` when the compatibility class is PATCH or MINOR.
        - ``compatibility_class`` (str): ``"PATCH"``, ``"MINOR"``, ``"BREAKING"``,
          or ``"UNKNOWN"`` if the manifest could not be loaded.
        - ``actions_required`` (list[str]): Human actions required before merging.
        - ``pinned_version`` (str): The target (pinned) contract version.
        - ``from_version`` (str): Echo of the supplied source version.
    """
    try:
        manifest = load_manifest(manifest_path)
    except (FileNotFoundError, ImportError) as exc:
        logger.warning("Cannot load contract manifest (%s) — defaulting to BREAKING class.", exc)
        return {
            "safe": False,
            "compatibility_class": "UNKNOWN",
            "actions_required": [
                "Manifest could not be loaded — verify upgrade safety manually.",
            ],
            "pinned_version": "UNKNOWN",
            "from_version": from_version,
        }

    pinned_version: str = manifest.get("schema_version", "UNKNOWN")
    compat_class: str = manifest.get("compatibility_class", "UNKNOWN")
    breaking_from: Dict[str, Any] = manifest.get("breaking_changes_from", {})

    actions: List[str] = []

    if compat_class == "PATCH":
        safe = True
    elif compat_class == "MINOR":
        safe = True
        actions.append("Human review required before merging (MINOR contract change).")
    else:
        # BREAKING or UNKNOWN
        safe = False
        actions.append("Human review required (BREAKING or unknown compatibility class).")
        actions.append("Attach and complete the migration checklist before merging.")
        if from_version in breaking_from:
            detail = breaking_from[from_version]
            actions.append(f"Breaking changes from {from_version}: {detail}")

    return {
        "safe": safe,
        "compatibility_class": compat_class,
        "actions_required": actions,
        "pinned_version": pinned_version,
        "from_version": from_version,
    }
