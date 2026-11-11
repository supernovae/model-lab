"""Run-seed boundaries, with no model, tokenizer, corpus or device work."""

import json
from pathlib import Path

import pytest
import yaml

from klab.config.loading import load_config
from klab.config.models import RunConfig
from klab.derivation import derive_config, receipt_path
from klab.lab_mode import run_try
from klab.training.manifest import config_sha256
from klab.try_config import TryConfig

FIXTURE = Path(__file__).parent / "fixtures/contracts/run_config_v2.json"


def _raw(seed=7, engine="pytorch"):
    raw = json.loads(FIXTURE.read_text())
    raw["seed"] = seed
    raw["runtime"].update(engine=engine, backend="metal" if engine == "mlx" else "cpu")
    return raw


def _write(path, raw):
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.mark.parametrize("engine", ["pytorch", "mlx"])
@pytest.mark.parametrize("seed", [-1, 2**32])
def test_invalid_seed_loading_and_derivation_have_no_outputs(tmp_path, engine, seed):
    source = _write(tmp_path / "source.yaml", _raw(seed, engine))
    with pytest.raises(ValueError, match="seed"):
        load_config(source)
    source = _write(source, _raw(7, engine))
    output = tmp_path / "derived.yaml"
    with pytest.raises(ValueError, match="seed"):
        derive_config(source, output, {"seed": seed})
    assert not output.exists()
    assert not receipt_path(output).exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["source.yaml"]


@pytest.mark.parametrize("engine", ["pytorch", "mlx"])
@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_boundary_loading_derivation_and_override_preserve_seed(tmp_path, engine, seed):
    source = _write(tmp_path / "source.yaml", _raw(seed, engine))
    assert load_config(source).seed == seed
    output = tmp_path / "derived.yaml"
    derive_config(source, output, {"seed": seed})
    assert load_config(output).seed == seed
    assert TryConfig(seed=seed).model_dump(mode="json")["seed"] == seed


@pytest.mark.parametrize(
    "seed,expected",
    [
        (0, "a8f9c6751cbfdc257d8fa1d58b484c681cdae8002b69c9990a78251fc2eee16f"),
        (2**32 - 1, "46428116b9c174c4239802f76ae45d9405d0f701326f9d07c8b6f9427d39fe61"),
    ],
)
def test_valid_seed_identity_matches_pre_fix(seed, expected):
    config = RunConfig.model_validate(_raw(seed))
    assert config_sha256(config.model_dump(mode="json")) == expected


@pytest.mark.parametrize("seed", [-1, 2**32])
@pytest.mark.parametrize("location", ["baseline", "candidate", "delta", "override"])
def test_try_rejects_invalid_seed_before_artifacts(
    tmp_path, monkeypatch, seed, location
):
    from klab import resource_envelope

    baseline = _write(
        tmp_path / "base.yaml", _raw(seed if location == "baseline" else 7)
    )
    candidate = _write(
        tmp_path / "candidate.yaml", _raw(seed if location == "candidate" else 7)
    )
    if location == "delta":
        _write(candidate, {"lab_delta_version": 1, "set": {"seed": seed}})
    monkeypatch.setattr(
        resource_envelope,
        "resolve_lab_try_limits",
        lambda *a, **kw: pytest.fail("resource preparation reached"),
    )
    with pytest.raises(ValueError, match="seed"):
        run_try(
            candidate,
            baseline,
            work_dir=tmp_path / "work",
            seed=seed if location == "override" else None,
            train_fn=lambda *a, **kw: pytest.fail("training reached"),
        )
    assert not (tmp_path / "work").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["base.yaml", "candidate.yaml"]


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_try_boundary_override_supersedes_delta_before_setup(
    tmp_path, monkeypatch, seed
):
    from klab import resource_envelope

    baseline = _write(tmp_path / "base.yaml", _raw())
    candidate = _write(
        tmp_path / "delta.yaml", {"lab_delta_version": 1, "set": {"seed": -1}}
    )

    class Validated(Exception):
        pass

    def stop_before_setup(*a, **kw):
        raise Validated

    monkeypatch.setattr(resource_envelope, "resolve_lab_try_limits", stop_before_setup)
    with pytest.raises(Validated):
        run_try(
            candidate,
            baseline,
            work_dir=tmp_path / "work",
            seed=seed,
            train_fn=lambda: None,
        )
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_try_rejects_merged_baseline_seed_before_artifacts(tmp_path, monkeypatch, seed):
    from klab import resource_envelope

    raw = _raw()
    del raw["seed"]
    baseline = tmp_path / "base.yaml"
    baseline.write_text(f"<<: &defaults {{seed: {seed}}}\n" + yaml.safe_dump(raw))
    candidate = _write(tmp_path / "candidate.yaml", _raw())
    monkeypatch.setattr(
        resource_envelope,
        "resolve_lab_try_limits",
        lambda *a, **kw: pytest.fail("resource preparation reached"),
    )
    with pytest.raises(ValueError, match="seed"):
        run_try(candidate, baseline, work_dir=tmp_path / "work", train_fn=lambda: None)
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_try_merged_baseline_uses_effective_valid_seed(tmp_path, monkeypatch, seed):
    from klab import resource_envelope

    raw = _raw(seed)
    baseline = tmp_path / "base.yaml"
    baseline.write_text("<<: &defaults {seed: -1}\n" + yaml.safe_dump(raw))
    assert load_config(baseline).seed == seed
    candidate = _write(tmp_path / "candidate.yaml", _raw())

    class Validated(Exception):
        pass

    def stop_before_setup(*a, **kw):
        raise Validated

    monkeypatch.setattr(resource_envelope, "resolve_lab_try_limits", stop_before_setup)
    with pytest.raises(Validated):
        run_try(candidate, baseline, work_dir=tmp_path / "work", train_fn=lambda: None)
    assert not (tmp_path / "work").exists()


@pytest.mark.parametrize("seed,expected", [(True, 1), ("7", 7), (7.0, 7)])
@pytest.mark.parametrize("suffix", ["yaml", "json"])
def test_legacy_seed_coercion_is_preserved_across_callers(
    tmp_path, seed, expected, suffix
):
    """LAB-67: stricter future contracts must not silently narrow RunConfig v2."""
    raw = _raw(seed)
    typed = RunConfig.model_validate(raw)
    assert typed.seed == expected
    assert config_sha256(typed.model_dump(mode="json")) == config_sha256(
        RunConfig.model_validate(_raw(expected)).model_dump(mode="json")
    )
    source = tmp_path / f"source.{suffix}"
    source.write_text(json.dumps(raw) if suffix == "json" else yaml.safe_dump(raw))
    assert load_config(source).seed == expected
    output = tmp_path / "derived.yaml"
    derive_config(source, output, {"seed": seed})
    assert load_config(output).seed == expected
    assert (
        json.loads(receipt_path(output).read_text())["declared_delta"]["seed"] == seed
    )
    with pytest.raises(ValueError, match="seed"):
        TryConfig(seed=seed)


@pytest.mark.parametrize("seed", [True, "7", 7.0])
def test_strict_try_override_rejected_before_setup(tmp_path, monkeypatch, seed):
    from klab import resource_envelope

    monkeypatch.setattr(
        resource_envelope,
        "resolve_lab_try_limits",
        lambda *a, **kw: pytest.fail("resource preparation reached"),
    )
    with pytest.raises(ValueError, match="seed"):
        run_try(
            tmp_path / "delta.yaml",
            tmp_path / "base.yaml",
            work_dir=tmp_path / "work",
            seed=seed,
        )
    assert not list(tmp_path.iterdir())
