"""Post-training: chat templates, objectives, preference pairs and export.

The staged tests run the shipped ``examples/post-training`` project with
smaller budgets, so the example and its documentation stay runnable.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import sysconfig
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from pydantic import ValidationError

from klab.chat_template import BUILTIN, ChatTemplate, resolve
from klab.config.loading import load_config, load_tokenizer_config
from klab.config.models import AttentionConfig, RunConfig
from klab.data.packing import PairedBlockDataset, TokenBlockDataset, prepare_data
from klab.data.tokenizer import load_tokenizer, train_tokenizer
from klab.evaluation.chat import chat_turn
from klab.evaluation.generation import generate_with_token_ids
from klab.evaluation.inference import load_run
from klab.lab_mode import current_inputs, run_try
from klab.training.objectives import read_checkpoint
from klab.training.trainer import train
from klab.training.weight_export import export_weights
from klab.training.weight_import import import_weights

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/post-training"
_CACHED_TOKENIZER: tuple[bytes, bytes] | None = None

_OBJECTIVE_PACKAGE = '''
from pydantic import BaseModel, ConfigDict

from klab.objectives import Objective
from klab.training.objectives import Loss, TrainingObjective, cross_entropy_sum


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confidence_penalty: float = 0.1


class Humble(TrainingObjective):
    """Cross-entropy plus a penalty on confident next-token distributions."""

    def __init__(self, penalty):
        self.penalty = penalty

    def loss(self, forward, batch, frozen):
        logits = forward(batch.inputs).float()
        supervised = batch.targets != -100
        log_probs = logits[supervised].log_softmax(-1)
        entropy = -(log_probs.exp() * log_probs).sum(-1)
        return Loss(
            cross_entropy_sum(logits, batch.targets) - self.penalty * entropy.sum(),
            {"entropy": entropy.detach().mean()},
        )


OBJECTIVE = Objective("toy_humble", lambda params: Humble(params.confidence_penalty), Params)
'''


@pytest.fixture(scope="module", autouse=True)
def toy_objective(tmp_path_factory: pytest.TempPathFactory):
    """An objective that reaches Lab only through its entry point."""
    root = tmp_path_factory.mktemp("site")
    (root / "toy_objectives.py").write_text(_OBJECTIVE_PACKAGE)
    info = root / "toy_objectives-0.1.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: toy-objectives\nVersion: 0.1\n"
    )
    (info / "entry_points.txt").write_text(
        "[klab.objectives]\ntoy_humble = toy_objectives:OBJECTIVE\n"
    )
    sys.path.insert(0, str(root))
    yield
    sys.path.remove(str(root))
    sys.modules.pop("toy_objectives", None)


def project(root: Path) -> Path:
    """Copy the example and train its tokenizer (cached per process)."""
    global _CACHED_TOKENIZER
    target = root / "project"
    shutil.copytree(EXAMPLE, target)
    tokenizer = target / "work/tokenizer"
    if _CACHED_TOKENIZER is None:
        path = train_tokenizer(load_tokenizer_config(target / "tokenizer.yaml"))
        _CACHED_TOKENIZER = (
            path.read_bytes(),
            (tokenizer / "tokenizer_manifest.json").read_bytes(),
        )
    else:
        tokenizer.mkdir(parents=True)
        (tokenizer / "tokenizer.json").write_bytes(_CACHED_TOKENIZER[0])
        (tokenizer / "tokenizer_manifest.json").write_bytes(_CACHED_TOKENIZER[1])
    return target


def stage(root: Path, name: str, steps: int, **training: object) -> RunConfig:
    """Load one of the example's stage configs with a smaller budget."""
    path = root / f"{name}.yaml"
    document = yaml.safe_load(path.read_text())
    document["training"].update(max_steps=steps, **training)
    document["optimizer"]["warmup_steps"] = 2
    document["evaluation"]["every_steps"] = steps
    document["checkpoint"]["every_steps"] = steps
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return load_config(path)


def metric(run: RunConfig, run_id: str, name: str) -> list[float]:
    with sqlite3.connect(run.logging.root_dir / "experiments.sqlite3") as database:
        return [
            value
            for (value,) in database.execute(
                "SELECT value FROM metrics WHERE run_id = ? AND name = ? ORDER BY step",
                (run_id, name),
            )
        ]


def events(run: RunConfig, run_id: str, kind: str) -> list[dict]:
    with sqlite3.connect(run.logging.root_dir / "experiments.sqlite3") as database:
        return [
            json.loads(payload)
            for (payload,) in database.execute(
                "SELECT payload_json FROM events WHERE run_id = ? AND kind = ?",
                (run_id, kind),
            )
        ]


@pytest.mark.parametrize("name", sorted(BUILTIN))
def test_templates_render_the_prompt_their_exported_jinja_renders(name: str) -> None:
    jinja2 = pytest.importorskip("jinja2")
    template = BUILTIN[name]
    messages = [
        {"role": "system", "content": " Be brief. "},
        {"role": "user", "content": "One?"},
        {"role": "assistant", "content": "Yes."},
        {"role": "user", "content": "Two?"},
    ]
    rendered = jinja2.Environment().from_string(template.to_jinja())
    assert rendered.render(
        messages=messages, add_generation_prompt=True
    ) == template.prompt(
        [("user", "One?"), ("assistant", "Yes.")], "Two?", system=" Be brief. "
    )
    # A complete conversation is the prompt without its final generation prompt.
    assert rendered.render(messages=messages[:3], add_generation_prompt=False) == (
        template.prompt([], "One?", system="Be brief.")
        + template.answer_lead
        + "Yes."
        + template.answer_tail
    )


def test_declared_templates_must_open_replies_with_their_generation_prompt() -> None:
    declared = {
        "name": "bracketed",
        "system": "[SYS] {content}\n",
        "user": "[USER] {content}\n",
        "assistant": "[BOT] {content}\n",
        "generation_prompt": "[BOT]",
        "stop": ["\n[USER]"],
    }
    template = resolve(declared)
    assert template.prompt([], "Hi") == "[USER] Hi\n[BOT]"
    assert (template.answer_lead, template.answer_tail) == (" ", "\n")
    with pytest.raises(ValidationError, match="start with generation_prompt"):
        ChatTemplate.model_validate({**declared, "generation_prompt": "[ASSISTANT]"})
    with pytest.raises(ValidationError, match="exactly once"):
        ChatTemplate.model_validate({**declared, "user": "[USER]\n"})
    with pytest.raises(ValueError, match="built in"):
        resolve({**declared, "name": "chatml"})
    with pytest.raises(ValueError, match="unknown chat template"):
        resolve("vicuna")


def test_objective_settings_refuse_runs_that_lack_what_the_objective_needs(
    tmp_path: Path,
) -> None:
    root = project(tmp_path)

    def variant(name: str, **training: object) -> dict:
        document = yaml.safe_load((root / f"{name}.yaml").read_text())
        document["tokenizer"]["path"] = str(root / "work/tokenizer/tokenizer.json")
        for key in ("train_path", "validation_path", "cache_dir"):
            document["dataset"][key] = str(root / document["dataset"][key])
        document["training"] = {**document["training"], **training}
        return document

    without_parent = variant("3-prefer")
    del without_parent["training"]["parent_checkpoint"]
    with pytest.raises(ValidationError, match=r"requires training\.parent_checkpoint"):
        RunConfig.model_validate(without_parent)
    with pytest.raises(ValidationError, match="requires dataset.source=local_pref"):
        RunConfig.model_validate(
            variant(
                "1-instruct",
                parent_checkpoint="work/runs/instruct",
                objective={"kind": "preference"},
            )
        )
    with pytest.raises(ValidationError, match="teacher_checkpoint"):
        RunConfig.model_validate(variant("1-instruct", objective={"kind": "distill"}))
    with pytest.raises(ValidationError, match="unknown objective kind"):
        RunConfig.model_validate(variant("1-instruct", objective={"kind": "reinforce"}))
    with pytest.raises(ValidationError, match="Extra inputs|extra_forbidden"):
        RunConfig.model_validate(
            variant(
                "2-distill",
                objective={
                    "kind": "distill",
                    "teacher_checkpoint": "work/runs/instruct",
                    "params": {"alpha": 0.5},
                },
            )
        )
    # Preference data with the default objective tunes on the chosen replies.
    plain = variant("3-prefer")
    del plain["training"]["objective"]
    assert RunConfig.model_validate(plain).training.objective.kind == "next_token"


def test_preference_pairs_pack_as_rows_every_reader_can_use(tmp_path: Path) -> None:
    root = project(tmp_path)
    run = stage(root, "3-prefer", 4)
    tokenizer = load_tokenizer(run.tokenizer.path)
    data = prepare_data(run, tokenizer)
    width = run.training.seq_len
    pairs = PairedBlockDataset(data.train, width, data.train_supervision, pad=0)
    assert len(pairs) == 160

    record = json.loads((root / "data/pairs-train.jsonl").read_text().splitlines()[0])
    inputs, targets, *_ = pairs.numpy_microblock(0)
    assert inputs.shape == targets.shape == (2, width)
    template = run.dataset.template

    def supervised_text(row: int) -> str:
        return tokenizer.decode(
            [int(token) for token in targets[row] if token != -100],
            skip_special_tokens=False,
        )

    # Each row supervises exactly its own reply, the closing marker and EOS.
    assert supervised_text(0) == record["chosen"] + template.answer_tail + "<eos>"
    assert supervised_text(1) == record["rejected"] + template.answer_tail + "<eos>"
    prompt_tokens = int(np.argmax(targets[0] != -100)) + 1
    assert np.array_equal(inputs[0, :prompt_tokens], inputs[1, :prompt_tokens])

    # A next-token reader sees the chosen replies only: rejected rows carry no
    # targets, so tuning or scoring on this data never rewards a rejected reply.
    plain = TokenBlockDataset(data.train, width, supervision=data.train_supervision)
    assert len(plain) == len(pairs)
    assert np.array_equal(plain.numpy_block(0)[1], targets[0])


@pytest.fixture
def instructed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, RunConfig]:
    """The example project with its instruction stage trained."""
    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "work"))
    root = project(tmp_path)
    run = stage(root, "1-instruct", 60)
    assert train(run, run_id="instruct") == "instruct"
    return root, run


def test_every_stage_lowers_its_objective_and_the_result_chats_and_exports(
    instructed: tuple[Path, RunConfig],
) -> None:
    root, instruct = instructed
    runs = instruct.logging.root_dir
    tokenizer = load_tokenizer(instruct.tokenizer.path)
    template = instruct.dataset.template
    # The tokenizer holds the template's markers as single tokens.
    assert template.missing_special_tokens(tokenizer) == ()
    validation = metric(instruct, "instruct", "validation/loss")
    assert validation[-1] < validation[0] / 2

    # Distillation: a smaller model moves toward the teacher's distribution.
    distill = stage(root, "2-distill", 60)
    train(distill, run_id="distill")
    divergence = metric(distill, "distill", "objective/teacher_kl")
    assert divergence[-1] < divergence[0] / 2
    assert metric(distill, "distill", "objective/teacher_agreement")[-1] > 0.3
    held_out = metric(distill, "distill", "validation/loss")
    assert held_out[-1] < held_out[0] / 2
    (recorded,) = events(distill, "distill", "training_objective")
    assert recorded["kind"] == "distill"
    assert recorded["teacher"]["run_id"] == "instruct"

    # Preference: stopped part-way and resumed, which reloads the reference.
    prefer = stage(root, "3-prefer", 40)
    train(prefer, run_id="prefer-first", stop_after_step=10)
    train(
        prefer,
        run_id="prefer",
        resume=runs / "prefer-first/checkpoints/latest.json",
    )
    assert metric(prefer, "prefer-first", "objective/preference_accuracy")[0] < 0.75
    assert metric(prefer, "prefer", "objective/preference_accuracy")[-1] > 0.75
    pair_loss = metric(prefer, "prefer-first", "validation/loss") + metric(
        prefer, "prefer", "validation/loss"
    )
    # An untouched policy scores log 2 against its own reference.
    assert pair_loss[0] == pytest.approx(0.6931, abs=1e-3)
    assert pair_loss[-1] < 0.35
    (recorded,) = events(prefer, "prefer", "training_objective")
    assert recorded["reference"]["run_id"] == "instruct"

    # The result loads and chats through the template it was trained with.
    loaded = load_run("prefer", runs)
    prompt, reply = chat_turn(
        loaded.model,
        loaded.tokenizer,
        [],
        "The printer is unplugged. What should I do?",
        loaded.config.model.max_seq_len,
        16,
        loaded.device,
        template=loaded.config.dataset.template,
    )
    assert prompt.endswith("<|im_start|>assistant\n")
    assert reply and "<|im_end|>" not in reply

    # Export, then import the exported directory back to identical weights.
    receipt = export_weights(runs / "prefer", root / "export")
    exported = root / "export"
    assert receipt["checkpoint_sha256"] == loaded.identity["checkpoint_sha256"]
    assert json.loads((exported / "config.json").read_text())["eos_token_id"] == [
        3,
        tokenizer.token_to_id("<|im_end|>"),
    ]
    assert (
        json.loads((exported / "tokenizer_config.json").read_text())["chat_template"]
        == template.to_jinja()
    )
    provenance = root / "provenance.json"
    provenance.write_text(
        json.dumps({"format_version": 1, "source": "export test", "license": "CC0-1.0"})
    )
    imported = import_weights(
        exported,
        runs / "imported",
        instruct.model_copy(
            update={
                "tokenizer": instruct.tokenizer.model_copy(
                    update={"path": exported / "tokenizer.json"}
                )
            }
        ),
        source_format="hf_llama_safetensors",
        source_tokenizer=exported / "tokenizer.json",
        provenance=provenance,
    )
    _, snapshot, _ = read_checkpoint(imported.checkpoint, role="imported")
    restored, original = snapshot.model, loaded.model.state_dict()
    assert restored.keys() == original.keys()
    assert all(torch.equal(restored[name], original[name]) for name in original)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        export_weights(runs / "prefer", exported)


def test_try_compares_two_preference_settings_from_the_same_parent(
    instructed: tuple[Path, RunConfig], tmp_path: Path
) -> None:
    root, _ = instructed
    stage(root, "3-prefer", 12)
    delta = root / "gentler.yaml"
    delta.write_text(
        yaml.safe_dump(
            {
                "question": "Does a tighter hold on the parent change the chosen replies' loss?",
                "set": {"training.objective.params.beta": 3.0},
            }
        )
    )
    record, _ = run_try(
        delta, root / "3-prefer.yaml", work_dir=tmp_path / "work", probe_tier="none"
    )
    assert record["status"] == "completed"
    baseline, candidate = record["arms"]["baseline"], record["arms"]["candidate"]
    # Both arms score the chosen replies of the same held-out pairs.
    assert baseline["heldout"]["valid_targets"] == candidate["heldout"]["valid_targets"]
    assert record["comparison"]["verdict"] != "NOT_COMPARABLE"
    # The parent is one of the inputs a reusable baseline is keyed on.
    inputs = current_inputs(load_config(root / "3-prefer.yaml"))
    assert len(inputs["checkpoints"]["parent"]) == 64


def test_an_objective_from_another_package_trains_and_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "work"))
    root = project(tmp_path)
    run = stage(
        root,
        "1-instruct",
        6,
        objective={"kind": "toy_humble", "params": {"confidence_penalty": 0.2}},
    )
    assert run.training.objective.params == {"confidence_penalty": 0.2}
    train(run, run_id="humble")
    entropy = metric(run, "humble", "objective/entropy")
    assert entropy and all(value > 0 for value in entropy)
    (recorded,) = events(run, "humble", "training_objective")
    assert recorded["source"] == "toy-objectives 0.1"


# Greedy decoding of an export in an interpreter that cannot import Lab. Each
# path reports its tokens and the logits every step chose from.
_EXPORTED_GREEDY = r"""
import importlib, importlib.util, json, sys

site_packages, directory, prompt, steps, use_transformers = sys.argv[1:]
# Torch and transformers, without the .pth file that puts Lab's source on the path.
sys.path.append(site_packages)
result = {"klab_importable": importlib.util.find_spec("klab") is not None}
import torch

prompt, steps = json.loads(prompt), int(steps)
config = json.loads(open(f"{directory}/config.json").read())
stops = config["eos_token_id"]
stops = set(stops if isinstance(stops, list) else [stops])
blocked = json.loads(open(f"{directory}/generation_config.json").read())[
    "suppress_tokens"
]

# The export as a plain package: Lab's decoder and cache, no transformers.
package = importlib.util.module_from_spec(
    importlib.util.spec_from_loader("exported", loader=None, is_package=True)
)
package.__path__ = [directory]
sys.modules["exported"] = package
loader = importlib.import_module("exported.klab_model_export_loader")
model = loader.load_model(directory)
limit, ids, cache = model.config.max_seq_len, list(prompt), None
tokens, scores = [], []
with torch.inference_mode():
    for _ in range(steps):
        if cache is None or cache.length >= cache.capacity:
            logits, cache = model.forward_cached(
                torch.tensor([ids[-limit:]]), cache_capacity=limit
            )
        else:
            logits, cache = model.forward_cached(torch.tensor([ids[-1:]]), cache=cache)
        scores.append(logits[0, -1].tolist())
        allowed = logits[0, -1].float().clone()
        allowed[blocked] = -torch.inf
        token = int(allowed.argmax())
        if token in stops:
            break
        tokens.append(token)
        ids.append(token)
result["package"] = {"tokens": tokens, "logits": scores}

if use_transformers == "1":
    from transformers import AutoModelForCausalLM, AutoTokenizer

    result["chat_template"] = AutoTokenizer.from_pretrained(directory).chat_template
    hf = AutoModelForCausalLM.from_pretrained(directory, trust_remote_code=True)
    output = hf.generate(
        torch.tensor([prompt]),
        max_new_tokens=steps,
        do_sample=False,
        output_logits=True,
        return_dict_in_generate=True,
    )
    tokens = output.sequences[0, len(prompt) :].tolist()
    if tokens and tokens[-1] in stops:
        tokens.pop()
    result["transformers"] = {
        "tokens": tokens,
        "logits": [step[0].tolist() for step in output.logits],
    }
print(json.dumps(result))
"""


def test_a_local_selected_ngram_run_exports_and_generates_without_lab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_training import config as tiny_config

    base = tiny_config(tmp_path)
    run = base.model_copy(
        update={
            "model": base.model.model_copy(
                update={
                    "memory": "ngram",
                    "memory_table_size": 31,
                    "memory_ngram_size": 2,
                    "memory_dim": 8,
                    # Untied, an untrained head does not just repeat the prompt.
                    "tie_embeddings": False,
                }
            ),
            "attention": AttentionConfig(
                kind="local_selected",
                block_size=4,
                window_size=4,
                selected_blocks=1,
                latent_dim=8,
                selector="learned",
            ),
            "training": base.training.model_copy(
                update={"max_steps": 3, "max_tokens": 96}
            ),
        }
    )
    run_id = train(run, run_id="sparse-memory")
    runs = run.logging.root_dir
    loaded = load_run(run_id, runs, None, "cpu")
    seen: list[torch.Tensor] = []
    forward_cached = loaded.model.forward_cached

    def recording(*args: object, **kwargs: object) -> object:
        logits, cache = forward_cached(*args, **kwargs)
        seen.append(logits[0, -1])
        return logits, cache

    monkeypatch.setattr(loaded.model, "forward_cached", recording)
    prompt, steps = "a b a b a", 40
    # 40 tokens past a 32-token context: the cache is rebuilt along the way.
    _, expected = generate_with_token_ids(
        loaded.model,
        loaded.tokenizer,
        prompt,
        run.model.max_seq_len,
        steps,
        loaded.device,
    )

    exported = tmp_path / "export"
    receipt = export_weights(runs / run_id, exported)
    assert receipt["format"] == "hf_remote_code"
    source = json.loads((exported / "config.json").read_text())["klab_source"]
    assert source["checkpoint_sha256"] == loaded.identity["checkpoint_sha256"]

    use_transformers = importlib.util.find_spec("transformers") is not None
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    (isolated / "greedy.py").write_text(_EXPORTED_GREEDY)
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(isolated / "greedy.py"),
            sysconfig.get_paths()["purelib"],
            str(exported),
            json.dumps(loaded.tokenizer.encode(prompt, add_special_tokens=False).ids),
            str(steps),
            "1" if use_transformers else "0",
        ],
        cwd=isolated,
        env={**os.environ, "HF_HOME": str(tmp_path / "hf"), "HF_HUB_OFFLINE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["klab_importable"] is False

    def same_decoding(path: dict) -> None:
        assert path["tokens"] == expected
        torch.testing.assert_close(torch.tensor(path["logits"]), torch.stack(seen))

    same_decoding(result["package"])
    pytest.importorskip("transformers")
    same_decoding(result["transformers"])
    assert result["chat_template"] == run.dataset.template.to_jinja()


def test_byte_memory_is_refused_before_anything_is_written(tmp_path: Path) -> None:
    from test_training import config as tiny_config

    base = tiny_config(tmp_path)
    run = base.model_copy(
        update={
            "model": base.model.model_copy(
                update={
                    "memory": "byte",
                    "memory_table_size": 64,
                    "memory_dim": 8,
                    "memory_ngram_size": 5,
                }
            ),
            "training": base.training.model_copy(
                update={"max_steps": 3, "max_tokens": 96}
            ),
        }
    )
    run_id = train(run, run_id="byte-memory")
    with pytest.raises(ValueError, match="byte memory"):
        export_weights(run.logging.root_dir / run_id, tmp_path / "export")
    assert not (tmp_path / "export").exists()
