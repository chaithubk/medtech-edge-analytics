"""Tests for the contract compatibility utility (src/utils/contract_compat.py).

Covers:
- parse_semver: valid and invalid inputs.
- is_compatible_version: same version, MINOR/PATCH upgrade, MAJOR change,
  legacy two-part format.
- load_manifest: file found, file missing, yaml unavailable (mocked).
- check_upgrade_safety: PATCH, MINOR, BREAKING, UNKNOWN (manifest unloadable).
"""

import pathlib

import pytest
import yaml

from src.utils.contract_compat import (
    PINNED_CONTRACT_VERSION,
    check_upgrade_safety,
    is_compatible_version,
    load_manifest,
    parse_semver,
)

# ---------------------------------------------------------------------------
# parse_semver
# ---------------------------------------------------------------------------


class TestParseSemver:
    def test_valid_three_part(self):
        assert parse_semver("2.0.0") == (2, 0, 0)

    def test_valid_minor_patch(self):
        assert parse_semver("2.1.3") == (2, 1, 3)

    def test_valid_large_numbers(self):
        assert parse_semver("10.20.300") == (10, 20, 300)

    def test_invalid_two_part(self):
        with pytest.raises(ValueError, match="Invalid SemVer"):
            parse_semver("2.0")

    def test_invalid_four_part(self):
        with pytest.raises(ValueError, match="Invalid SemVer"):
            parse_semver("2.0.0.0")

    def test_invalid_non_numeric(self):
        with pytest.raises(ValueError, match="Invalid SemVer"):
            parse_semver("two.zero.zero")

    def test_invalid_empty(self):
        with pytest.raises(ValueError, match="Invalid SemVer"):
            parse_semver("")


# ---------------------------------------------------------------------------
# is_compatible_version
# ---------------------------------------------------------------------------


class TestIsCompatibleVersion:
    def test_exact_match(self):
        """Exact version match is always compatible."""
        assert is_compatible_version("2.0.0") is True

    def test_minor_upgrade_compatible(self):
        """MINOR upgrade within same MAJOR is compatible."""
        assert is_compatible_version("2.1.0") is True

    def test_patch_upgrade_compatible(self):
        """PATCH upgrade is compatible."""
        assert is_compatible_version("2.0.1") is True

    def test_higher_minor_compatible(self):
        """Significantly higher MINOR is still compatible (backward-compatible additions)."""
        assert is_compatible_version("2.99.0") is True

    def test_major_upgrade_incompatible(self):
        """Different MAJOR version is always incompatible (BREAKING)."""
        assert is_compatible_version("3.0.0") is False

    def test_lower_major_incompatible(self):
        """Lower MAJOR version is also incompatible."""
        assert is_compatible_version("1.0.0") is False

    def test_legacy_two_part_accepted(self):
        """Legacy two-part '2.0' format is accepted during migration window."""
        assert is_compatible_version("2.0") is True

    def test_legacy_two_part_higher_minor_accepted(self):
        """Legacy two-part '2.1' format is also accepted."""
        assert is_compatible_version("2.1") is True

    def test_legacy_two_part_major_one_rejected(self):
        """Legacy '1.0' two-part is rejected (different MAJOR)."""
        assert is_compatible_version("1.0") is False

    def test_none_version_incompatible(self):
        """None / missing version is not compatible."""
        assert is_compatible_version(None) is False  # type: ignore[arg-type]

    def test_empty_string_incompatible(self):
        """Empty string is not compatible."""
        assert is_compatible_version("") is False

    def test_numeric_version_incompatible(self):
        """Numeric version values are not compatible."""
        assert is_compatible_version(2) is False  # type: ignore[arg-type]

    def test_custom_pinned_version(self):
        """is_compatible_version respects a custom pinned_version argument."""
        assert is_compatible_version("3.0.0", pinned_version="3.0.0") is True
        assert is_compatible_version("2.0.0", pinned_version="3.0.0") is False


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------


class TestLoadManifest:
    def test_loads_vendored_manifest(self):
        """The real vendored manifest loads without error."""
        manifest = load_manifest()
        assert isinstance(manifest, dict)

    def test_manifest_has_required_keys(self):
        """Vendored manifest contains all governance metadata fields."""
        manifest = load_manifest()
        for key in (
            "schema_version",
            "pinned_tag",
            "pinned_commit",
            "contract_repo",
            "canonical_schema_path",
            "compatibility_class",
        ):
            assert key in manifest, f"Manifest missing key: '{key}'"

    def test_manifest_schema_version_matches_pinned(self):
        """Manifest schema_version must match PINNED_CONTRACT_VERSION."""
        manifest = load_manifest()
        assert manifest["schema_version"] == PINNED_CONTRACT_VERSION

    def test_custom_path(self, tmp_path):
        """load_manifest accepts an explicit path argument."""
        custom = tmp_path / "test-manifest.yml"
        custom.write_text(
            "schema_version: '9.9.9'\npinned_tag: v9.9.9\n"
            "contract_repo: test/repo\npinned_commit: '0000'\n"
            "canonical_schema_path: schemas/test.json\n"
            "compatibility_class: PATCH\n",
            encoding="utf-8",
        )
        manifest = load_manifest(custom)
        assert manifest["schema_version"] == "9.9.9"

    def test_missing_manifest_raises(self, tmp_path):
        """FileNotFoundError raised when the manifest file does not exist."""
        with pytest.raises(FileNotFoundError, match="Contract manifest not found"):
            load_manifest(tmp_path / "nonexistent.yml")

    def test_env_var_overrides_default(self, tmp_path, monkeypatch):
        """MEDTECH_VITALS_MANIFEST env var overrides the default path."""
        custom = tmp_path / "env-manifest.yml"
        custom.write_text(
            "schema_version: '1.2.3'\npinned_tag: v1.2.3\n"
            "contract_repo: test/repo\npinned_commit: 'abc'\n"
            "canonical_schema_path: schemas/test.json\n"
            "compatibility_class: MINOR\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("MEDTECH_VITALS_MANIFEST", str(custom))
        manifest = load_manifest()
        assert manifest["schema_version"] == "1.2.3"


# ---------------------------------------------------------------------------
# check_upgrade_safety
# ---------------------------------------------------------------------------


class TestCheckUpgradeSafety:
    def _make_manifest(
        self,
        tmp_path: pathlib.Path,
        compat_class: str,
        breaking_from: dict | None = None,
        schema_version: str = "2.1.0",
    ) -> pathlib.Path:
        data = {
            "schema_version": schema_version,
            "pinned_tag": f"v{schema_version}",
            "pinned_commit": "abc123",
            "contract_repo": "test/repo",
            "canonical_schema_path": "schemas/vitals/vitals.schema.json",
            "compatibility_class": compat_class,
            "breaking_changes_from": breaking_from or {},
            "vendored_at": "2026-01-01",
        }
        p = tmp_path / "manifest.yml"
        p.write_text(yaml.dump(data), encoding="utf-8")
        return p

    def test_patch_upgrade_safe(self, tmp_path):
        manifest_path = self._make_manifest(tmp_path, "PATCH", schema_version="2.0.1")
        result = check_upgrade_safety("2.0.0", manifest_path)
        assert result["safe"] is True
        assert result["compatibility_class"] == "PATCH"
        assert result["actions_required"] == []

    def test_minor_upgrade_safe_requires_review(self, tmp_path):
        manifest_path = self._make_manifest(tmp_path, "MINOR")
        result = check_upgrade_safety("2.0.0", manifest_path)
        assert result["safe"] is True
        assert result["compatibility_class"] == "MINOR"
        assert len(result["actions_required"]) > 0
        assert any("review" in a.lower() for a in result["actions_required"])

    def test_breaking_upgrade_not_safe(self, tmp_path):
        manifest_path = self._make_manifest(
            tmp_path,
            "BREAKING",
            breaking_from={"2.0.0": "Removed field 'legacy_field'"},
            schema_version="3.0.0",
        )
        result = check_upgrade_safety("2.0.0", manifest_path)
        assert result["safe"] is False
        assert result["compatibility_class"] == "BREAKING"
        assert any("migration" in a.lower() for a in result["actions_required"])

    def test_breaking_with_specific_version_includes_detail(self, tmp_path):
        """When from_version appears in breaking_changes_from, details are included."""
        manifest_path = self._make_manifest(
            tmp_path,
            "BREAKING",
            breaking_from={"2.0.0": "Field 'hr' renamed to 'heart_rate'"},
            schema_version="3.0.0",
        )
        result = check_upgrade_safety("2.0.0", manifest_path)
        assert result["safe"] is False
        detail_present = any("heart_rate" in a or "hr" in a for a in result["actions_required"])
        assert detail_present

    def test_missing_manifest_returns_unknown(self, tmp_path):
        """When manifest cannot be loaded the result is safe=False, class=UNKNOWN."""
        result = check_upgrade_safety("2.0.0", tmp_path / "nonexistent.yml")
        assert result["safe"] is False
        assert result["compatibility_class"] == "UNKNOWN"

    def test_result_includes_version_fields(self, tmp_path):
        """Result always includes pinned_version and from_version for traceability."""
        manifest_path = self._make_manifest(tmp_path, "PATCH", schema_version="2.0.1")
        result = check_upgrade_safety("2.0.0", manifest_path)
        assert result["from_version"] == "2.0.0"
        assert result["pinned_version"] == "2.0.1"
