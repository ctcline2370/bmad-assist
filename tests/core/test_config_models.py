"""Tests for configuration Pydantic models.

These tests verify acceptance criteria for Story 1.2 (AC1-AC5):
- AC1: Config Model Exists with Required Sections
- AC2: Provider Configuration Model
- AC3: Invalid Configuration Raises ValidationError
- AC4: Nested Model Validation
- AC5: Default Values Work Correctly

Note: AC6 (Singleton Pattern) and edge cases are in test_config_models_singleton.py.
Extracted from test_config.py as part of Story 1.8 (Test Suite Refactoring).
"""

import pytest
from pydantic import ValidationError

from bmad_assist.core.config import (
    BmadPathsConfig,
    Config,
    MasterProviderConfig,
    MultiProviderConfig,
    PowerPromptConfig,
    ProviderConfig,
    ToolGuardConfig,
)

# === AC1: Config Model Structure ===


class TestConfigModelStructure:
    """Tests for AC1: Config model contains all required sections."""

    def test_config_has_providers_section(self) -> None:
        """Config model has providers section."""
        config = Config(
            providers=ProviderConfig(master=MasterProviderConfig(provider="claude", model="opus_4"))
        )
        assert hasattr(config, "providers")
        assert isinstance(config.providers, ProviderConfig)

    def test_config_has_power_prompts_section(self) -> None:
        """Config model has power_prompts section."""
        config = Config(
            providers=ProviderConfig(master=MasterProviderConfig(provider="claude", model="opus_4"))
        )
        assert hasattr(config, "power_prompts")
        assert isinstance(config.power_prompts, PowerPromptConfig)

    def test_config_has_state_path_field(self) -> None:
        """Config model has state_path field (str | None with default None)."""
        config = Config(
            providers=ProviderConfig(master=MasterProviderConfig(provider="claude", model="opus_4"))
        )
        assert hasattr(config, "state_path")
        # Default is None, use get_state_path() for resolved path
        assert config.state_path is None

    def test_config_has_bmad_paths_section(self) -> None:
        """Config model has bmad_paths section."""
        config = Config(
            providers=ProviderConfig(master=MasterProviderConfig(provider="claude", model="opus_4"))
        )
        assert hasattr(config, "bmad_paths")
        assert isinstance(config.bmad_paths, BmadPathsConfig)

    def test_all_fields_have_type_hints(self) -> None:
        """All Config fields have type hints (verified via Pydantic model_fields)."""
        # Pydantic requires type hints - this test ensures they exist
        assert "providers" in Config.model_fields
        assert "power_prompts" in Config.model_fields
        assert "state_path" in Config.model_fields
        assert "bmad_paths" in Config.model_fields


class TestToolGuardConfig:
    """Tests for tool guard hardening configuration."""

    def test_tool_guard_accepts_tenant_budget_and_diagnostics_fields(self) -> None:
        """Project configs can set tenant limits and diagnostics toggles."""
        config = Config.model_validate(
            {
                "providers": {
                    "master": {
                        "provider": "codex",
                        "model": "gpt-5",
                    }
                },
                "tool_guard": {
                    "max_total_calls": 350,
                    "max_interactions_per_file": 25,
                    "max_calls_per_minute": 120,
                    "per_tenant_max_tokens": 20_000,
                    "tenant_credit_window": "per_run",
                    "tenant_circuit_breaker_threshold": 0.8,
                    "diagnostics_enabled": True,
                    "diagnostics_enabled_env": "BMAD_ASSIST_DIAGNOSTICS_ENABLED",
                },
            }
        )

        assert config.tool_guard.max_total_calls == 350
        assert config.tool_guard.per_tenant_max_tokens == 20_000
        assert config.tool_guard.tenant_credit_window == "per_run"
        assert config.tool_guard.tenant_circuit_breaker_threshold == 0.8
        assert config.tool_guard.diagnostics_enabled is True
        assert config.tool_guard.diagnostics_enabled_env == "BMAD_ASSIST_DIAGNOSTICS_ENABLED"

    def test_tool_guard_new_fields_have_safe_defaults(self) -> None:
        """Tenant-aware guard fields have production-safe defaults."""
        guard = ToolGuardConfig()

        assert guard.per_tenant_max_tokens == 20_000
        assert guard.tenant_credit_window == "per_run"
        assert guard.tenant_circuit_breaker_threshold == 0.8
        assert guard.diagnostics_enabled is False
        assert guard.diagnostics_enabled_env == "BMAD_ASSIST_DIAGNOSTICS_ENABLED"


# === AC2: Provider Configuration Model ===


class TestProviderConfigModel:
    """Tests for AC2: Provider configuration model structure."""

    def test_master_provider_config_required_fields(self) -> None:
        """MasterProviderConfig requires provider and model fields."""
        config = MasterProviderConfig(provider="claude", model="opus_4")
        assert config.provider == "claude"
        assert config.model == "opus_4"

    def test_master_provider_config_settings_optional(self) -> None:
        """MasterProviderConfig settings is optional and defaults to None."""
        config = MasterProviderConfig(provider="claude", model="opus_4")
        assert config.settings is None
        assert config.model_name is None
        assert config.display_model == "opus_4"

    def test_master_provider_with_settings(self) -> None:
        """MasterProviderConfig accepts settings and model_name."""
        config = MasterProviderConfig(
            provider="claude",
            model="opus_4",
            settings="./provider-configs/master.json",
            model_name="glm-4.7",
        )
        assert config.settings == "./provider-configs/master.json"
        assert config.model_name == "glm-4.7"
        assert config.display_model == "glm-4.7"

    def test_multi_provider_config_same_structure(self) -> None:
        """MultiProviderConfig has same structure as MasterProviderConfig."""
        config = MultiProviderConfig(
            provider="gemini", model="gemini_2_5_pro", settings="./gemini.json"
        )
        assert config.provider == "gemini"
        assert config.model == "gemini_2_5_pro"
        assert config.settings == "./gemini.json"
        assert config.display_model == "gemini_2_5_pro"  # No model_name, uses model

    def test_provider_config_multi_list(self) -> None:
        """ProviderConfig.multi accepts list of MultiProviderConfig."""
        config = ProviderConfig(
            master=MasterProviderConfig(provider="claude", model="opus_4"),
            multi=[
                MultiProviderConfig(provider="gemini", model="gemini_2_5_pro"),
                MultiProviderConfig(provider="codex", model="o3"),
            ],
        )
        assert len(config.multi) == 2
        assert config.multi[0].provider == "gemini"
        assert config.multi[1].provider == "codex"


# === AC3: Invalid Configuration Raises ValidationError ===


class TestInvalidConfigurationErrors:
    """Tests for AC3: Invalid configuration raises ValidationError."""

    def test_missing_provider_field_raises_error(self) -> None:
        """Missing required 'provider' field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MasterProviderConfig(model="opus_4")  # type: ignore[call-arg]
        error_str = str(exc_info.value)
        assert "provider" in error_str

    def test_missing_model_field_raises_error(self) -> None:
        """Missing required 'model' field raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MasterProviderConfig(provider="claude")  # type: ignore[call-arg]
        error_str = str(exc_info.value)
        assert "model" in error_str

    def test_wrong_type_for_provider_raises_error(self) -> None:
        """Wrong type for provider raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MasterProviderConfig(provider=123, model="opus_4")  # type: ignore[arg-type]
        error_str = str(exc_info.value)
        assert "provider" in error_str

    def test_wrong_type_for_model_raises_error(self) -> None:
        """Wrong type for model raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MasterProviderConfig(provider="claude", model=123)  # type: ignore[arg-type]
        error_str = str(exc_info.value)
        assert "model" in error_str
        assert "str" in error_str.lower()

    def test_missing_master_in_providers_raises_error(self) -> None:
        """Missing required 'master' field in ProviderConfig raises error."""
        with pytest.raises(ValidationError) as exc_info:
            ProviderConfig()  # type: ignore[call-arg]
        error_str = str(exc_info.value)
        assert "master" in error_str

    def test_missing_providers_in_config_raises_error(self) -> None:
        """Missing required 'providers' field in Config raises error."""
        with pytest.raises(ValidationError) as exc_info:
            Config()  # type: ignore[call-arg]
        error_str = str(exc_info.value)
        assert "providers" in error_str

    def test_error_message_is_human_readable(self) -> None:
        """ValidationError message is human-readable with field info."""
        with pytest.raises(ValidationError) as exc_info:
            MasterProviderConfig(provider="claude", model=123)  # type: ignore[arg-type]
        errors = exc_info.value.errors()
        assert len(errors) > 0
        # Each error should have location and message
        error = errors[0]
        assert "loc" in error
        assert "msg" in error
        assert "type" in error


# === AC4: Nested Model Validation ===


class TestNestedValidationErrors:
    """Tests for AC4: Nested validation error includes full path."""

    def test_nested_validation_error_path_via_dict(self) -> None:
        """Nested validation error from dict includes path to field."""
        with pytest.raises(ValidationError) as exc_info:
            Config.model_validate(
                {
                    "providers": {
                        "master": {"provider": "claude", "model": 123}  # model wrong type
                    }
                }
            )
        errors = exc_info.value.errors()
        # Find the error for the model field
        model_errors = [e for e in errors if "model" in str(e.get("loc", []))]
        assert len(model_errors) > 0
        # Location should include the nested path
        loc = model_errors[0]["loc"]
        assert "providers" in loc or "master" in loc or "model" in loc

    def test_deeply_nested_error_includes_full_path(self) -> None:
        """Deeply nested validation error shows full dotted path."""
        with pytest.raises(ValidationError) as exc_info:
            Config.model_validate(
                {
                    "providers": {
                        "master": {"provider": "claude", "model": "opus_4"},
                        "multi": [
                            {"provider": "gemini", "model": 456}  # model wrong type
                        ],
                    }
                }
            )
        errors = exc_info.value.errors()
        # Should have error in multi list
        assert len(errors) > 0
        error_str = str(exc_info.value)
        assert "multi" in error_str or "model" in error_str

    def test_config_model_validate_with_nested_dict(self) -> None:
        """Config.model_validate works with properly nested dict."""
        config = Config.model_validate(
            {"providers": {"master": {"provider": "claude", "model": "opus_4"}}}
        )
        assert config.providers.master.provider == "claude"
        assert config.providers.master.model == "opus_4"


# === AC5: Default Values Work Correctly ===


class TestDefaultValues:
    """Tests for AC5: Default values work correctly."""

    def test_default_state_path(self) -> None:
        """Default state_path is None; use get_state_path() for resolved path."""
        from bmad_assist.core.state import get_state_path

        config = Config(
            providers=ProviderConfig(master=MasterProviderConfig(provider="claude", model="opus_4"))
        )
        # Default is None, get_state_path() provides resolved default
        assert config.state_path is None

        # get_state_path returns expanded default
        resolved = get_state_path(config)
        assert "~" not in str(resolved)
        assert str(resolved).endswith(".bmad-assist/state.yaml")

    def test_default_multi_is_empty_list(self) -> None:
        """Default multi providers is empty list."""
        config = ProviderConfig(master=MasterProviderConfig(provider="claude", model="opus_4"))
        assert config.multi == []
        assert isinstance(config.multi, list)

    def test_default_power_prompts_set_name_is_none(self) -> None:
        """Default power_prompts.set_name is None."""
        config = Config(
            providers=ProviderConfig(master=MasterProviderConfig(provider="claude", model="opus_4"))
        )
        assert config.power_prompts.set_name is None

    def test_default_power_prompts_variables_is_empty_dict(self) -> None:
        """Default power_prompts.variables is empty dict."""
        config = Config(
            providers=ProviderConfig(master=MasterProviderConfig(provider="claude", model="opus_4"))
        )
        assert config.power_prompts.variables == {}
        assert isinstance(config.power_prompts.variables, dict)

    def test_default_bmad_paths_all_none(self) -> None:
        """Default bmad_paths fields are all None."""
        config = Config(
            providers=ProviderConfig(master=MasterProviderConfig(provider="claude", model="opus_4"))
        )
        assert config.bmad_paths.prd is None
        assert config.bmad_paths.architecture is None
        assert config.bmad_paths.epics is None
        assert config.bmad_paths.stories is None

    def test_minimal_valid_config_with_defaults(self) -> None:
        """Minimal configuration with only required fields is valid."""
        from bmad_assist.core.state import get_state_path

        config = Config(
            providers=ProviderConfig(master=MasterProviderConfig(provider="claude", model="opus_4"))
        )
        # All defaults should be applied (state_path is None, resolved via get_state_path)
        assert config.state_path is None
        resolved = get_state_path(config)
        assert "~" not in str(resolved)
        assert str(resolved).endswith(".bmad-assist/state.yaml")
        assert config.power_prompts.set_name is None
        assert config.power_prompts.variables == {}
        assert config.bmad_paths.prd is None
        assert config.providers.multi == []

    def test_default_settings_is_none(self) -> None:
        """Default settings for providers is None."""
        master = MasterProviderConfig(provider="claude", model="opus_4")
        multi = MultiProviderConfig(provider="gemini", model="gemini_2_5_pro")
        assert master.settings is None
        assert multi.settings is None
        assert master.model_name is None
        assert multi.model_name is None
