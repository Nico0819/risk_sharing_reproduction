"""Offline regressions for project-owned LLM settings and framework aliases."""

from __future__ import annotations

import json
import os
import socket
import sys

import pytest

from custom.config import RuntimeConfigError, load_runtime_config


PROJECT_VALUES = {
    "LLM_API_KEY": "fake-project-key-for-tests",
    "LLM_BASE_URL": "https://project-provider.invalid/v1",
    "LLM_MODEL": "project-test-model",
}
ALIASES = {
    f"{prefix}_{suffix}": source
    for prefix in ("AGENTSOCIETY_LLM", "AGENTSOCIETY_CODER_LLM")
    for suffix, source in (
        ("API_KEY", "LLM_API_KEY"),
        ("API_BASE", "LLM_BASE_URL"),
        ("MODEL", "LLM_MODEL"),
    )
}
THINKING_NAMES = ("AGENTSOCIETY_LLM_THINKING", "AGENTSOCIETY_CODER_LLM_THINKING")
BODY_NAMES = ("AGENTSOCIETY_LLM_EXTRA_BODY", "AGENTSOCIETY_CODER_LLM_EXTRA_BODY")
EFFORT_NAMES = ("AGENTSOCIETY_LLM_REASONING_EFFORT", "AGENTSOCIETY_CODER_LLM_REASONING_EFFORT")
ROLE_SETTINGS = {
    "AGENTSOCIETY_LLM_THINKING": "on",
    "AGENTSOCIETY_LLM_EXTRA_BODY": '{"thinking":{"type":"enabled"}}',
    "AGENTSOCIETY_CODER_LLM_THINKING": "off",
    "AGENTSOCIETY_CODER_LLM_EXTRA_BODY": '{"thinking":{"type":"disabled"}}',
    **{name: "" for name in EFFORT_NAMES},
}


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch):
    """Keep tests independent of the caller's real credentials and settings."""

    framework = sys.modules.get("agentsociety2.config.config")
    if framework is not None:
        for alias in (*ALIASES, *ROLE_SETTINGS):
            attribute = alias.removeprefix("AGENTSOCIETY_")
            monkeypatch.setattr(framework.Config, attribute, getattr(framework.Config, attribute))
    for name in (
        *PROJECT_VALUES, *ALIASES, *ROLE_SETTINGS,
        "AGENTSOCIETY_LLM_REQUEST_TIMEOUT",
    ):
        monkeypatch.delenv(name, raising=False)


def write_env(tmp_path, values=None):
    path = tmp_path / ".env"
    values = PROJECT_VALUES if values is None else values
    path.write_text("".join(f"{name}={value}\n" for name, value in values.items()), encoding="utf-8")
    return path


def assert_aliases(values=PROJECT_VALUES):
    assert {name: os.environ.get(name) for name in ALIASES} == {
        name: values[source] for name, source in ALIASES.items()
    }


def stale_aliases(monkeypatch):
    values = {name: f"fake-old-{number}" for number, name in enumerate(ALIASES)}
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return values


def test_project_config_replaces_stale_framework_aliases_without_exposing_keys(tmp_path, monkeypatch, capsys):
    old = stale_aliases(monkeypatch)
    path = write_env(tmp_path)
    before = path.read_bytes()

    config = load_runtime_config("llm", env_path=path)

    assert_aliases()
    assert config.api_key == PROJECT_VALUES["LLM_API_KEY"]
    assert config.base_url == PROJECT_VALUES["LLM_BASE_URL"]
    assert config.model == PROJECT_VALUES["LLM_MODEL"]
    assert config.safe_status()["api_key"] == "configured"
    assert config.thinking_disabled is False
    assert config.router_thinking_disabled is True
    assert config.safe_status()["thinking"] == "enabled"
    assert config.safe_status()["participant_thinking"] == "enabled"
    assert config.safe_status()["router_thinking"] == "disabled"
    output = capsys.readouterr()
    diagnostics = output.out + output.err + repr(config) + json.dumps(config.safe_status())
    assert PROJECT_VALUES["LLM_API_KEY"] not in diagnostics
    assert old["AGENTSOCIETY_LLM_API_KEY"] not in diagnostics
    assert path.read_bytes() == before


@pytest.mark.parametrize("old_value", ["", "   "])
def test_empty_framework_aliases_are_filled(tmp_path, monkeypatch, old_value):
    for name in ALIASES:
        monkeypatch.setenv(name, old_value)

    load_runtime_config("llm", env_path=write_env(tmp_path))

    assert_aliases()


def test_explicit_project_environment_still_wins_over_dotenv(tmp_path, monkeypatch):
    values = {
        "LLM_API_KEY": "fake-process-key-for-tests",
        "LLM_BASE_URL": "https://process-provider.invalid/v1",
        "LLM_MODEL": "process-test-model",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, f"  {value}  ")
    stale_aliases(monkeypatch)

    config = load_runtime_config("llm", env_path=write_env(tmp_path))

    assert_aliases(values)
    assert (config.api_key, config.base_url, config.model) == tuple(values.values())


@pytest.mark.parametrize("missing", PROJECT_VALUES)
def test_framework_aliases_do_not_supply_missing_project_settings(tmp_path, monkeypatch, missing):
    old = stale_aliases(monkeypatch)
    values = {name: value for name, value in PROJECT_VALUES.items() if name != missing}

    with pytest.raises(RuntimeConfigError, match=f"missing or placeholder LLM configuration: {missing}"):
        load_runtime_config("llm", env_path=write_env(tmp_path, values))

    assert {name: os.environ[name] for name in ALIASES} == old


@pytest.mark.parametrize("placeholder", ["", "changeme", "<your-api-key>"])
def test_invalid_project_key_leaves_framework_settings_unchanged(tmp_path, monkeypatch, placeholder):
    old = stale_aliases(monkeypatch)
    values = {**PROJECT_VALUES, "LLM_API_KEY": placeholder}

    with pytest.raises(RuntimeConfigError, match="missing or placeholder"):
        load_runtime_config("llm", env_path=write_env(tmp_path, values))

    assert {name: os.environ[name] for name in ALIASES} == old


def test_reloading_config_maps_new_project_values_instead_of_rejecting_old_aliases(tmp_path):
    path = write_env(tmp_path)
    first = load_runtime_config("llm", env_path=path)
    assert load_runtime_config("llm", env_path=path) == first
    next_values = {
        "LLM_API_KEY": "fake-next-project-key",
        "LLM_BASE_URL": "https://next-provider.invalid/v1",
        "LLM_MODEL": "next-test-model",
    }
    write_env(tmp_path, next_values)

    second = load_runtime_config("llm", env_path=path)

    assert_aliases(next_values)
    assert (second.api_key, second.base_url, second.model) == tuple(next_values.values())


def test_scripted_constructor_defaults_do_not_block_a_later_llm_configuration(tmp_path):
    path = write_env(tmp_path)
    load_runtime_config("scripted", env_path=path)
    assert os.environ["AGENTSOCIETY_LLM_API_KEY"] == "scripted-local-only"

    config = load_runtime_config("llm", env_path=path)

    assert_aliases()
    assert config.backend == "llm"


@pytest.mark.parametrize("old_mode", ["off", "on", "unrelated-old-setting"])
def test_project_assigns_distinct_role_modes_despite_stale_aliases(tmp_path, monkeypatch, old_mode):
    stale_aliases(monkeypatch)
    for name in THINKING_NAMES:
        monkeypatch.setenv(name, old_mode)
    for name in BODY_NAMES:
        monkeypatch.setenv(name, "old non-JSON body that must not reach the provider")
    for name in EFFORT_NAMES:
        monkeypatch.setenv(name, "none")
    path = write_env(tmp_path)
    before = path.read_bytes()

    load_runtime_config("llm", env_path=path)

    assert_aliases()
    assert {name: os.environ.get(name) for name in ROLE_SETTINGS} == ROLE_SETTINGS
    assert path.read_bytes() == before


def test_missing_credentials_do_not_partially_change_thinking_modes(tmp_path, monkeypatch):
    stale = {name: "fake-old-value" for name in ROLE_SETTINGS}
    for name, value in stale.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeConfigError, match="missing or placeholder"):
        load_runtime_config("llm", env_path=write_env(tmp_path, {}))

    assert {name: os.environ.get(name) for name in ROLE_SETTINGS} == stale


def test_scripted_mode_continues_to_preserve_existing_framework_aliases(tmp_path, monkeypatch):
    old = stale_aliases(monkeypatch)
    old_thinking = {name: "fake-old-value" for name in ROLE_SETTINGS}
    for name, value in old_thinking.items():
        monkeypatch.setenv(name, value)

    config = load_runtime_config("scripted", env_path=write_env(tmp_path, {}))

    assert {name: os.environ[name] for name in ALIASES} == old
    assert {name: os.environ[name] for name in ROLE_SETTINGS} == old_thinking
    assert config.backend == "scripted"
    assert config.api_key == ""
    assert config.safe_status()["participant_thinking"] == "disabled"
    assert config.safe_status()["router_thinking"] == "disabled"


@pytest.mark.parametrize("role", ["default", "coder"])
def test_existing_framework_config_is_refreshed_before_building_real_client(tmp_path, monkeypatch, role):
    """Exercise the installed factory without initializing a network runtime."""

    def unexpected_network(*args, **kwargs):
        pytest.fail("A configuration-only check must not access the network")

    monkeypatch.setattr(socket.socket, "connect", unexpected_network)
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    for name, value in PROJECT_VALUES.items():
        monkeypatch.setenv(name, value)
    # Some framework dependencies try to discover a dotenv file at import
    # time. This test exclusively uses the explicit temporary file below.
    import dotenv
    import dotenv.main

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(dotenv.main, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(dotenv, "dotenv_values", lambda *args, **kwargs: {})
    monkeypatch.setattr(dotenv.main, "dotenv_values", lambda *args, **kwargs: {})
    monkeypatch.chdir(tmp_path)
    path = write_env(tmp_path)
    load_runtime_config("llm", env_path=path)

    from agentsociety2.config import Config, build_client_for_role

    # Model a framework imported earlier with a different configuration. Use
    # monkeypatch so every cached class attribute is restored after this test.
    for alias in (*ALIASES, *ROLE_SETTINGS):
        monkeypatch.setattr(Config, alias.removeprefix("AGENTSOCIETY_"), "fake-cached-old-value")
    stale_aliases(monkeypatch)

    load_runtime_config("llm", env_path=path)
    client = build_client_for_role(role)

    connection_matches = (
        client.api_key, client.base_url, client.model_name,
    ) == tuple(PROJECT_VALUES.values())
    assert connection_matches, "The constructed client must use the fictional project settings"
    assert client.model_type == role
    expected_state = "enabled" if role == "default" else "disabled"
    assert client.thinking.policy == ("on" if role == "default" else "off")
    assert client.thinking.request_kwargs() == {"extra_body": {"thinking": {"type": expected_state}}}
    for alias, value in ROLE_SETTINGS.items():
        assert getattr(Config, alias.removeprefix("AGENTSOCIETY_")) == value
    assert client._router is None
