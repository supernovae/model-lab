"""Strict, immutable schemas for Model Lab training and tokenizer jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

# Shared run/try bounds reject out-of-range seeds rather than wrapping them.
RunSeed = Annotated[int, Field(ge=0, le=2**32 - 1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MemoryTokenMapping(StrictModel):
    algorithm: Literal["identity", "deepseek_v1", "deepseek_safe_v1"]
    token_ids: tuple[Annotated[int, Field(strict=True, ge=0)], ...]
    vocab_size: int = Field(strict=True, gt=0)
    memory_vocab_size: int = Field(strict=True, gt=0)
    tokenizer_sha256: str
    mapping_sha256: str

    @model_validator(mode="after")
    def validate_mapping(self) -> MemoryTokenMapping:
        from klab.memory_tokenizer import validate_memory_token_map

        validate_memory_token_map(self.model_dump(mode="json"))
        return self


class ModelConfig(StrictModel):
    vocab_size: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int | None = None
    ffn_dim: int
    max_seq_len: int
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True
    ffn: Literal["dense", "moe"] = "dense"
    num_experts: int = 1
    experts_per_token: int = 1
    shared_expert: bool = False
    router_aux_loss_coefficient: float = Field(default=0.0, ge=0)
    memory: Literal["none", "ngram", "byte", "portable"] = "none"
    memory_injection: Literal["final", "embedding", "after_block"] = "final"
    memory_block_index: int | None = Field(
        default=None, strict=True, ge=0, exclude_if=lambda value: value is None
    )
    memory_block_indices: tuple[Annotated[int, Field(strict=True, ge=0)], ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    memory_variant: Literal["hidden_gate", "context_key"] = Field(
        default="hidden_gate", exclude_if=lambda value: value == "hidden_gate"
    )
    memory_conv_kernel_size: int = Field(
        default=0, strict=True, ge=0, le=16, exclude_if=lambda value: value == 0
    )
    memory_gate_transform: Literal["identity", "signed_sqrt"] = Field(
        default="identity", exclude_if=lambda value: value == "identity"
    )
    memory_addressing: Literal["legacy", "prime_xor_v1"] = Field(
        default="legacy", exclude_if=lambda value: value == "legacy"
    )
    memory_table_sizes: tuple[Annotated[int, Field(strict=True, gt=0)], ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    memory_hash_seed: int = Field(
        default=0, strict=True, ge=0, le=2**32 - 1, exclude_if=lambda value: value == 0
    )
    memory_token_mapping: MemoryTokenMapping | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    memory_padding_token_id: int = Field(
        default=0, strict=True, ge=0, exclude_if=lambda value: value == 0
    )
    memory_independent_init: bool = Field(
        default=False, strict=True, exclude_if=lambda value: not value
    )
    memory_table_size: int = 0
    memory_ngram_size: int = 0
    memory_dim: int = 0
    memory_package_path: Path | None = None
    memory_ngram_orders: tuple[int, ...] = ()
    memory_hash_heads: int = Field(default=1, gt=0)
    semantic_memory_dim: int | None = Field(default=None, gt=0)

    @property
    def memory_block_sites(self) -> tuple[int, ...]:
        return self.memory_block_indices or (
            () if self.memory_block_index is None else (self.memory_block_index,)
        )

    @model_serializer(mode="wrap")
    def _serialize_legacy_final(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        result = handler(self)
        if self.memory_injection == "final":
            result.pop("memory_injection", None)
        if self.num_kv_heads is None:
            result.pop("num_kv_heads", None)
        return result

    @model_validator(mode="after")
    def validate_dimensions(self) -> ModelConfig:
        values = (
            self.vocab_size,
            self.hidden_dim,
            self.num_layers,
            self.num_heads,
            self.ffn_dim,
            self.max_seq_len,
        )
        if any(value <= 0 for value in values):
            raise ValueError("model dimensions must be positive")
        if self.vocab_size < 260:
            raise ValueError("model.vocab_size must be at least 260")
        if self.hidden_dim % self.num_heads or (self.hidden_dim // self.num_heads) % 2:
            raise ValueError("model head dimension must be even and divide hidden_dim")
        if self.num_kv_heads is not None and (
            self.num_kv_heads <= 0
            or self.num_kv_heads > self.num_heads
            or self.num_heads % self.num_kv_heads
        ):
            raise ValueError(
                "model.num_kv_heads must be positive, no greater than num_heads, "
                "and divide num_heads"
            )
        if self.rms_norm_eps <= 0:
            raise ValueError("model.rms_norm_eps must be positive")
        if self.ffn == "dense" and (
            self.num_experts != 1
            or self.experts_per_token != 1
            or self.shared_expert
            or self.router_aux_loss_coefficient
        ):
            raise ValueError("dense model.ffn does not accept MoE settings")
        if self.ffn == "moe" and (
            self.num_experts < 2 or not 1 <= self.experts_per_token <= self.num_experts
        ):
            raise ValueError("invalid MoE expert configuration")
        if self.memory_block_indices and (
            self.memory_block_index is not None
            or tuple(sorted(set(self.memory_block_indices)))
            != self.memory_block_indices
        ):
            raise ValueError(
                "memory_block_indices must be sorted unique and excludes memory_block_index"
            )
        if self.memory_injection == "after_block":
            if self.memory != "ngram" or not self.memory_block_sites:
                raise ValueError(
                    "after_block requires token ngram memory and memory_block_index or memory_block_indices"
                )
            if max(self.memory_block_sites) >= self.num_layers:
                raise ValueError(
                    "memory_block_index must identify an existing decoder block"
                )
        elif self.memory_block_index is not None or self.memory_block_indices:
            raise ValueError("memory block indices require after_block injection")
        if self.memory_variant != "hidden_gate" and self.memory != "ngram":
            raise ValueError("context_key requires token ngram memory")
        if (
            self.memory_gate_transform != "identity"
            and self.memory_variant != "context_key"
        ):
            raise ValueError("gate transform requires context_key memory")
        if self.memory_conv_kernel_size and self.memory_variant != "context_key":
            raise ValueError("memory convolution requires context_key memory")
        if self.memory != "ngram" and (
            self.memory_addressing != "legacy"
            or self.memory_token_mapping is not None
            or self.memory_independent_init
            or self.memory_padding_token_id
        ):
            raise ValueError(
                "token addressing/mapping/independent initialization requires ngram memory"
            )
        if self.memory_addressing == "prime_xor_v1":
            from klab.address_hash import validate_engram_table_sizes

            streams = (len(self.memory_ngram_orders) or 1) * self.memory_hash_heads
            validate_engram_table_sizes(
                self.memory_table_sizes, streams * max(1, len(self.memory_block_sites))
            )
            if self.memory_table_sizes[0] != self.memory_table_size:
                raise ValueError("memory_table_size must equal first prime stream size")
        elif self.memory_table_sizes or self.memory_hash_seed:
            raise ValueError("per-stream sizes and hash seed require prime_xor_v1")
        if self.memory_padding_token_id >= self.vocab_size:
            raise ValueError("memory padding token must belong to vocabulary")
        if (
            self.memory_token_mapping is not None
            and self.memory_token_mapping.vocab_size != self.vocab_size
        ):
            raise ValueError("memory mapping vocabulary differs from model vocabulary")
        settings = (self.memory_table_size, self.memory_ngram_size, self.memory_dim)
        if self.memory == "none" and self.memory_injection != "final":
            raise ValueError("embedding memory injection requires enabled memory")
        if self.memory == "none" and (
            any(settings)
            or self.memory_package_path is not None
            or self.memory_ngram_orders
            or self.memory_hash_heads != 1
        ):
            raise ValueError("disabled model.memory requires zero memory settings")
        if self.memory != "none" and (
            self.memory_table_size <= 0
            or self.memory_ngram_size < 2
            or self.memory_dim <= 0
        ):
            raise ValueError(
                "enabled model.memory requires table, dimension, and ngram size"
            )
        if self.memory_ngram_orders and (
            any(order < 2 for order in self.memory_ngram_orders)
            or tuple(sorted(set(self.memory_ngram_orders))) != self.memory_ngram_orders
        ):
            raise ValueError(
                "model.memory_ngram_orders must be sorted unique values >= 2"
            )
        if self.memory == "byte" and self.memory_ngram_orders:
            raise ValueError("byte memory uses its configured raw-byte ngram size")
        if (self.memory == "portable") != (self.memory_package_path is not None):
            raise ValueError("portable memory requires exactly memory_package_path")
        return self


class TokenizerOrigin(StrictModel):
    """Explicit immutable origin of a tokenizer reused across frozen snapshots."""

    dataset: DatasetConfig | None = Field(default=None, exclude_if=lambda v: v is None)
    mixture_declaration_path: Path | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    mixture_declaration_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$", exclude_if=lambda v: v is None
    )
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_origin(self) -> TokenizerOrigin:
        if (self.dataset is None) == (self.mixture_declaration_path is None):
            raise ValueError(
                "tokenizer origin requires exactly one snapshot or mixture declaration"
            )
        if (self.mixture_declaration_path is None) != (
            self.mixture_declaration_sha256 is None
        ):
            raise ValueError("tokenizer origin mixture declaration requires its digest")
        if self.dataset is not None and self.dataset.source != "snapshot":
            raise ValueError("explicit tokenizer origin requires a frozen snapshot")
        return self


class TokenizerConfig(StrictModel):
    path: Path
    origin: TokenizerOrigin | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class DatasetConfig(StrictModel):
    source: Literal[
        "local_text",
        "local_chat",
        "local_token_mixture",
        "snapshot",
        "huggingface",
        "synthetic",
    ]
    # Hugging Face dataset repository, for source=huggingface.
    dataset_id: str | None = Field(default=None, exclude_if=lambda value: value is None)
    text_field: str = Field(
        default="text", min_length=1, exclude_if=lambda value: value == "text"
    )
    train_split: str = Field(
        default="train", min_length=1, exclude_if=lambda value: value == "train"
    )
    # Without a validation split, validation follows the training documents.
    validation_split: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    # Built-in offline generator, for source=synthetic.
    synthetic_task: Literal[
        "stories",
        "instruction_reference",
        "chat_recall",
        "engram_recall",
        "withheld_facts",
    ] = Field(default="stories", exclude_if=lambda value: value == "stories")
    revision: str | None = None
    dataset_config: str | None = None
    cache_dir: Path
    train_max_documents: int = Field(gt=0)
    validation_max_documents: int = Field(gt=0)
    train_max_tokens: int = Field(gt=0)
    validation_max_tokens: int = Field(gt=0)
    synthetic_seed: int = 42
    train_path: Path | None = None
    validation_path: Path | None = None
    source_manifest_path: Path | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    license: str | None = None
    allocation_manifest_path: Path | None = None
    corpus_release_path: Path | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    corpus_export_path: Path | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    mixture_declaration_path: Path | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    mixture_output_path: Path | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def validate_source(self) -> DatasetConfig:
        if self.source == "local_token_mixture":
            if (
                self.mixture_declaration_path is None
                or self.mixture_output_path is None
            ):
                raise ValueError(
                    "local_token_mixture requires mixture declaration and output"
                )
            if (
                self.corpus_release_path is not None
                or self.corpus_export_path is not None
            ):
                raise ValueError(
                    "local_token_mixture authenticates its own release closure"
                )
        elif (
            self.mixture_declaration_path is not None
            or self.mixture_output_path is not None
        ):
            raise ValueError("mixture paths require source=local_token_mixture")
        if self.allocation_manifest_path is not None and self.source != "local_chat":
            raise ValueError(
                "dataset.allocation_manifest_path requires source=local_chat"
            )
        if self.source_manifest_path is not None and self.source != "snapshot":
            raise ValueError("dataset.source_manifest_path requires source=snapshot")
        if self.source != "synthetic" and self.synthetic_task != "stories":
            raise ValueError("dataset.synthetic_task requires source=synthetic")
        if self.dataset_config is not None and self.source not in {
            "huggingface",
            "snapshot",
        }:
            raise ValueError(
                "dataset.dataset_config requires source=huggingface or snapshot"
            )
        if self.source != "huggingface" and (
            self.dataset_id is not None
            or self.validation_split is not None
            or self.text_field != "text"
            or self.train_split != "train"
        ):
            raise ValueError(
                "dataset_id, text_field and split names require source=huggingface"
            )
        if (self.corpus_release_path is None) != (self.corpus_export_path is None):
            raise ValueError(
                "corpus_release_path and corpus_export_path must be supplied together"
            )
        if self.corpus_release_path is not None:
            if self.source not in {"local_text", "local_chat"}:
                raise ValueError("corpus export requires local_text or local_chat")
            if (
                not self.revision
                or len(self.revision) != 64
                or any(char not in "0123456789abcdef" for char in self.revision)
            ):
                raise ValueError("corpus export requires full SHA-256 dataset.revision")
        if self.source == "snapshot":
            if (
                not self.revision
                or len(self.revision) not in {40, 64}
                or any(char not in "0123456789abcdef" for char in self.revision)
                or not self.license
                or not self.license.strip()
            ):
                raise ValueError(
                    "snapshot requires pinned revision and explicit license"
                )
            if any(
                path is None
                for path in (
                    self.train_path,
                    self.validation_path,
                    self.source_manifest_path,
                )
            ):
                raise ValueError(
                    f"{self.source} requires train_path, validation_path and source_manifest_path"
                )
            if self.train_path.resolve() == self.validation_path.resolve():
                raise ValueError(
                    f"{self.source} train and validation must be separate files"
                )
            if self.source_manifest_path.resolve() in {
                self.train_path.resolve(),
                self.validation_path.resolve(),
            }:
                raise ValueError(
                    f"{self.source} manifest must be separate from split files"
                )
            return self
        if self.source == "huggingface":
            if not self.dataset_id or not self.dataset_id.strip():
                raise ValueError("huggingface requires dataset.dataset_id")
            if not self.revision:
                raise ValueError("dataset.revision is required for remote datasets")
            if not self.license or not self.license.strip():
                raise ValueError(
                    "huggingface requires explicit dataset.license provenance"
                )
            if self.train_path is not None or self.validation_path is not None:
                raise ValueError("huggingface reads splits, not local paths")
            return self
        if self.source in {"local_chat", "local_text", "local_token_mixture"}:
            if self.train_path is None or self.validation_path is None:
                raise ValueError(
                    f"{self.source} requires train_path and validation_path"
                )
            if self.train_path.resolve() == self.validation_path.resolve():
                raise ValueError(
                    f"{self.source} training and validation must be separate files"
                )
            if not self.license or not self.license.strip():
                raise ValueError(
                    f"{self.source} requires explicit dataset.license provenance"
                )
        elif any(
            value is not None
            for value in (self.train_path, self.validation_path, self.license)
        ):
            raise ValueError(
                "train_path, validation_path and license are only for local, snapshot or huggingface sources"
            )
        return self


class ActivationCheckpointingConfig(StrictModel):
    enabled: bool = False
    strategy: Literal["transformer_block"] = "transformer_block"


class ActivationOffloadConfig(StrictModel):
    enabled: bool = False


class MemoryConfig(StrictModel):
    estimate_policy: Literal["enforce", "observe"] = Field(
        default="enforce", exclude_if=lambda value: value == "enforce"
    )
    policy: Literal["fast", "balanced", "low_memory", "max_fit"] = "balanced"
    max_device_memory_fraction: float = Field(default=0.90, gt=0, le=1)
    budget_bytes: int | None = Field(default=None, gt=0)
    activation_checkpointing: ActivationCheckpointingConfig = (
        ActivationCheckpointingConfig()
    )
    activation_offload: ActivationOffloadConfig = ActivationOffloadConfig()
    allowed_sequence_lengths: tuple[int, ...] = ()
    allowed_optimizers: tuple[Literal["adamw", "adafactor"], ...] = ()

    @model_validator(mode="after")
    def validate_lengths(self) -> MemoryConfig:
        if any(value <= 0 for value in self.allowed_sequence_lengths):
            raise ValueError("runtime.memory.allowed_sequence_lengths must be positive")
        return self


class RuntimeConfig(StrictModel):
    engine: Literal["pytorch", "mlx"] = "pytorch"
    backend: Literal["auto", "cpu", "mps", "cuda", "rocm", "xpu", "metal"] = "auto"
    device_index: int = Field(default=0, ge=0)
    secondary_device_indices: tuple[int, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )
    precision: Literal["auto", "fp32", "bf16", "fp16"] = "fp32"
    memory: MemoryConfig = MemoryConfig()
    compile_dynamic_shapes: bool = Field(
        default=False, exclude_if=lambda value: not value
    )
    compile_transformer_blocks: bool = Field(
        default=False, exclude_if=lambda value: not value
    )
    compile_output_head_loss: bool = Field(
        default=False, exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def validate_block_compilation(self) -> RuntimeConfig:
        if self.compile_dynamic_shapes and not (
            self.compile_transformer_blocks or self.compile_output_head_loss
        ):
            raise ValueError("dynamic compilation requires a compiled training kernel")
        if self.compile_transformer_blocks and (
            self.engine != "pytorch"
            or self.memory.activation_checkpointing.enabled
            or self.memory.activation_offload.enabled
        ):
            raise ValueError(
                "block compilation requires PyTorch without activation checkpointing or offload"
            )
        if self.compile_output_head_loss and (
            self.engine != "pytorch"
            or self.backend != "xpu"
            or self.precision != "bf16"
            or self.memory.activation_offload.enabled
        ):
            raise ValueError(
                "compiled output head and loss require PyTorch XPU BF16 "
                "without activation offload"
            )
        if self.secondary_device_indices:
            if self.engine != "pytorch" or self.backend in {"mps", "metal"}:
                raise ValueError(
                    "secondary_device_indices requires PyTorch on cpu, cuda, rocm, or xpu"
                )
            if any(idx < 0 for idx in self.secondary_device_indices):
                raise ValueError("secondary_device_indices must be non-negative")
            if len(set(self.secondary_device_indices)) != len(
                self.secondary_device_indices
            ):
                raise ValueError("secondary_device_indices must not contain duplicates")
            if (
                self.backend != "cpu"
                and self.device_index in self.secondary_device_indices
            ):
                raise ValueError(
                    "secondary_device_indices must not include the primary device_index"
                )
            if self.memory.activation_offload.enabled:
                raise ValueError(
                    "secondary_device_indices cannot be combined with activation offload"
                )
            if self.precision == "fp16":
                raise ValueError(
                    "secondary_device_indices cannot be combined with fp16: scaled "
                    "updates run on the primary device only; use bf16 or fp32"
                )
        return self


class TargetGroupTransition(StrictModel):
    """One explicit first-pass regrouping at a completed checkpoint boundary."""

    after_tokens: int = Field(strict=True, gt=0)
    previous_targets_per_update: int = Field(strict=True, gt=0)
    parent_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MatchedStoryConfig(StrictModel):
    """Shared document grouping; layout is the sole arm-specific setting."""

    layout: Literal["contiguous", "story"]
    targets_per_update: int = Field(default=4096, gt=0)
    unpadded_documents: bool = False
    document_batching: Literal["separate", "padded"] = Field(
        default="separate", exclude_if=lambda value: value == "separate"
    )
    padding_multiple: int = Field(
        default=128, gt=0, exclude_if=lambda value: value == 128
    )
    target_group_transition: TargetGroupTransition | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    max_padded_slots: int | None = Field(
        default=None, gt=0, exclude_if=lambda v: v is None
    )
    max_attention_cells: int | None = Field(
        default=None, gt=0, exclude_if=lambda v: v is None
    )
    frozen_update_ends: tuple[Annotated[int, Field(strict=True, gt=0)], ...] = Field(
        default=(), exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def validate_frozen_ends(self) -> MatchedStoryConfig:
        if self.target_group_transition is not None and (
            not self.unpadded_documents or self.layout != "story"
        ):
            raise ValueError("target regrouping requires whole source-order documents")
        if self.document_batching == "padded" and not self.unpadded_documents:
            raise ValueError(
                "padded document batching requires whole unpadded documents"
            )
        if tuple(sorted(set(self.frozen_update_ends))) != self.frozen_update_ends:
            raise ValueError("frozen update ends must be sorted and unique")
        return self


class TrainingConfig(StrictModel):
    stable_steps: int = Field(
        default=0, strict=True, ge=0, exclude_if=lambda value: value == 0
    )
    micro_batch_size: int = Field(gt=0)
    gradient_accumulation: int = Field(default=1, gt=0)
    seq_len: int = Field(gt=0)
    max_steps: int = Field(gt=0)
    max_tokens: int = Field(gt=0)
    grad_clip_norm: float = Field(default=1.0, gt=0)
    neural_loss_weight: float = Field(default=1.0, ge=0, le=1)
    deterministic: bool = True
    matched_stories: MatchedStoryConfig | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    trainable_parameters: tuple[str, ...] | None = None
    portability_manifest_path: Path | None = None

    @model_serializer(mode="wrap")
    def _serialize_optional_experiment_fields(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        result = handler(self)
        if self.trainable_parameters is None:
            result.pop("trainable_parameters", None)
        if self.portability_manifest_path is None:
            result.pop("portability_manifest_path", None)
        return result

    @model_validator(mode="after")
    def validate_trainable_parameters(self) -> TrainingConfig:
        names = self.trainable_parameters
        if names is not None and (
            not names
            or any(not name or not name.strip() for name in names)
            or len(names) != len(set(names))
        ):
            raise ValueError(
                "training.trainable_parameters must be a nonempty tuple of unique canonical names"
            )
        if (
            self.portability_manifest_path is not None
            and not str(self.portability_manifest_path).strip()
        ):
            raise ValueError("training.portability_manifest_path must not be blank")
        return self


class AdamWConfig(StrictModel):
    memory_table_lr_multiplier: float = Field(
        default=1.0, gt=0, allow_inf_nan=False, exclude_if=lambda value: value == 1.0
    )
    memory_table_weight_decay: float | None = Field(
        default=None, ge=0, allow_inf_nan=False, exclude_if=lambda value: value is None
    )
    name: Literal["adamw"] = "adamw"
    peak: float = Field(default=3e-4, gt=0)
    floor: float = Field(default=3e-5, ge=0)
    warmup_steps: int = Field(default=10, ge=0)
    decay_steps: int | None = Field(
        default=None, strict=True, gt=0, exclude_if=lambda value: value is None
    )
    weight_decay: float = Field(default=0.1, ge=0)
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = Field(default=1e-8, gt=0)
    state_offload: bool = False


class AdafactorConfig(StrictModel):
    name: Literal["adafactor"] = "adafactor"
    peak: float = Field(default=3e-4, gt=0)
    floor: float = Field(default=3e-5, ge=0)
    warmup_steps: int = Field(default=10, ge=0)
    weight_decay: float = Field(default=0.1, ge=0)
    beta2_decay: float = -0.8
    eps: tuple[float | None, float] = (None, 0.001)
    d: float = Field(default=1.0, gt=0)
    state_offload: bool = False


OptimizerConfig = Annotated[AdamWConfig | AdafactorConfig, Field(discriminator="name")]


class AttentionConfig(StrictModel):
    kind: Literal[
        "dense", "sliding_window", "mla", "block_sparse", "local_selected"
    ] = "dense"
    implementation: Literal["reference", "sdpa"] = Field(
        default="reference", exclude_if=lambda value: value == "reference"
    )
    rope_base: float = Field(default=10000.0, gt=0)
    window_size: int | None = Field(default=None, gt=0)
    latent_dim: int | None = Field(default=None, gt=0)
    block_size: int | None = Field(default=None, gt=0)
    selected_blocks: int | None = Field(default=None, gt=0)
    selector: Literal["mean_key", "learned"] = Field(
        default="mean_key", exclude_if=lambda value: value == "mean_key"
    )

    @model_validator(mode="after")
    def validate_kind(self) -> AttentionConfig:
        sparse = (
            self.window_size,
            self.latent_dim,
            self.block_size,
            self.selected_blocks,
        )
        if self.kind == "dense" and any(value is not None for value in sparse):
            raise ValueError("dense attention does not accept sparse settings")
        if self.kind == "sliding_window" and (
            self.window_size is None or any(value is not None for value in sparse[1:])
        ):
            raise ValueError("sliding_window attention requires only window_size")
        if self.kind == "mla" and (
            self.latent_dim is None
            or self.window_size is not None
            or self.block_size is not None
            or self.selected_blocks is not None
        ):
            raise ValueError("mla attention requires only latent_dim")
        if self.kind == "block_sparse" and (
            self.block_size is None
            or self.selected_blocks is None
            or self.window_size is not None
            or self.latent_dim is not None
        ):
            raise ValueError(
                "block_sparse attention requires block_size and selected_blocks"
            )
        if self.kind == "local_selected":
            if (
                self.block_size is None
                or self.selected_blocks is None
                or self.window_size is None
            ):
                raise ValueError(
                    "local_selected attention requires block_size, window_size "
                    "and selected_blocks"
                )
            if self.window_size % self.block_size:
                raise ValueError(
                    "local_selected window_size must be a multiple of block_size"
                )
            if self.implementation != "reference":
                raise ValueError(
                    "local_selected attention has one implementation; "
                    "leave attention.implementation unset"
                )
        if self.selector != "mean_key" and self.kind != "local_selected":
            raise ValueError("attention.selector requires kind=local_selected")
        if self.implementation == "sdpa" and self.kind not in {
            "dense",
            "sliding_window",
            "mla",
        }:
            raise ValueError(
                "SDPA implementation requires dense, sliding_window or mla attention"
            )
        return self


class FixedValidationPanel(StrictModel):
    """Immutable prepared selection, independent of any particular checkpoint."""

    name: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]
    prepared_path: Path
    prepared_manifest_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    tokenizer_file_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    tokenizer_canonical_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    split: Literal["validation"]
    sampling: Literal["first_eligible_blocks"]
    sequence_length: Annotated[int, Field(strict=True, gt=0)]
    blocks: Annotated[int, Field(strict=True, gt=0)]
    supervised_targets: Annotated[int, Field(strict=True, gt=0)]
    batch_size: Annotated[int, Field(strict=True, gt=0)]
    max_forward_positions: Annotated[int, Field(strict=True, gt=0)]

    @model_validator(mode="after")
    def budget(self) -> FixedValidationPanel:
        positions = self.blocks * self.sequence_length
        if (
            positions > self.max_forward_positions
            or self.supervised_targets > positions
        ):
            raise ValueError("panel selection exceeds its forward or target budget")
        return self


class EvaluationConfig(StrictModel):
    every_steps: int = Field(default=25, gt=0)
    max_batches: int = Field(default=16, gt=0)
    batch_size: int | None = Field(
        default=None, gt=0, exclude_if=lambda value: value is None
    )
    panels: tuple[FixedValidationPanel, ...] = Field(
        default=(), exclude_if=lambda value: not value
    )

    @property
    def batch_limit(self) -> int:
        """Actual fixed-panel batches, or the ordinary validation upper bound."""
        if self.panels:
            return sum(
                (p.blocks + p.batch_size - 1) // p.batch_size for p in self.panels
            )
        return self.max_batches

    @model_validator(mode="after")
    def unique_panels(self) -> EvaluationConfig:
        if len({p.name for p in self.panels}) != len(self.panels):
            raise ValueError("fixed validation requires distinct named panels")
        return self


class CheckpointMilestone(StrictModel):
    name: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]
    step: Annotated[int, Field(strict=True, ge=0)]
    committed_targets: Annotated[int, Field(strict=True, ge=0)]


class CheckpointRetention(StrictModel):
    milestones: tuple[CheckpointMilestone, ...] = Field(max_length=128)
    terminal_step: Annotated[int, Field(strict=True, gt=0)]
    terminal_targets: Annotated[int, Field(strict=True, gt=0)]
    max_generations: Annotated[int, Field(strict=True, gt=0, le=1024)]
    max_generation_bytes: Annotated[int, Field(strict=True, gt=0)]
    max_retained_bytes: Annotated[int, Field(strict=True, gt=0)]
    max_peak_bytes: Annotated[int, Field(strict=True, gt=0)]
    reserve_free_bytes: Annotated[int, Field(strict=True, ge=0)]

    @property
    def retained_upper(self) -> int:
        # Latest, previous, best and terminal may all be distinct after recovery.
        return len(self.milestones) + 4

    @model_validator(mode="after")
    def finite_plan(self) -> CheckpointRetention:
        if len({item.name for item in self.milestones}) != len(self.milestones):
            raise ValueError("checkpoint milestone names must be unique")
        if any(
            item.step > self.terminal_step
            or item.committed_targets > self.terminal_targets
            for item in self.milestones
        ):
            raise ValueError("checkpoint milestone exceeds terminal boundary")
        if (
            self.max_generations < self.retained_upper
            or self.max_retained_bytes < self.retained_upper * self.max_generation_bytes
            or self.max_peak_bytes
            < (self.retained_upper + 1) * self.max_generation_bytes
        ):
            raise ValueError(
                "retention limits cannot fit the conservative keep-set and write headroom"
            )
        return self


class CheckpointConfig(StrictModel):
    every_steps: int | None = Field(default=50, gt=0)
    every_tokens: int | None = Field(default=None, gt=0)
    every_minutes: float | None = Field(default=None, gt=0)
    keep_periodic: bool = True
    retention: CheckpointRetention | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    steps: tuple[Annotated[int, Field(strict=True, ge=0)], ...] = Field(
        default=(), exclude_if=lambda value: not value
    )

    @model_validator(mode="after")
    def validate_cadence(self) -> CheckpointConfig:
        if self.retention is not None and self.keep_periodic:
            raise ValueError("milestone retention requires keep_periodic=false")
        if tuple(sorted(set(self.steps))) != self.steps:
            raise ValueError("checkpoint steps must be strictly increasing and unique")
        if (
            self.every_steps is None
            and self.every_tokens is None
            and self.every_minutes is None
            and not self.steps
        ):
            raise ValueError("at least one checkpoint cadence is required")
        return self


class StagingConfig(StrictModel):
    smoke_steps: int = Field(default=2, gt=0)
    warmup_steps: int = Field(default=5, gt=0)


class LoggingConfig(StrictModel):
    root_dir: Path
    every_steps: int = Field(default=1, gt=0)
    architecture_diagnostics: Literal["scalar", "full"] = "scalar"


class RunConfig(StrictModel):
    schema_version: Literal[2]
    name: str = Field(min_length=1)
    seed: RunSeed
    runtime: RuntimeConfig = RuntimeConfig()
    model: ModelConfig
    tokenizer: TokenizerConfig
    dataset: DatasetConfig
    training: TrainingConfig
    optimizer: OptimizerConfig = AdamWConfig()
    attention: AttentionConfig = AttentionConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    staging: StagingConfig = StagingConfig()
    logging: LoggingConfig

    @model_validator(mode="after")
    def validate_cross_section(self) -> RunConfig:
        matched = self.training.matched_stories
        if (
            matched is not None
            and matched.unpadded_documents
            and (
                matched.layout != "story"
                or (
                    self.training.micro_batch_size != 1
                    and matched.document_batching != "padded"
                )
                or self.evaluation.batch_size not in {None, 1}
                or (
                    self.training.micro_batch_size > 1
                    and self.evaluation.batch_size != 1
                )
                or self.runtime.engine != "pytorch"
                or self.model.memory not in {"none", "ngram"}
            )
        ):
            raise ValueError(
                "unpadded documents require story layout, batch size one and PyTorch none/ngram memory"
            )
        if (
            matched is not None
            and matched.document_batching == "padded"
            and (
                self.attention.kind != "dense"
                or self.model.ffn != "dense"
                or self.training.gradient_accumulation != 1
                or self.dataset.allocation_manifest_path is not None
                or self.training.portability_manifest_path is not None
            )
        ):
            raise ValueError(
                "padded document batching requires dense attention/FFN, no sidecars and accumulation one"
            )
        if self.runtime.compile_output_head_loss and (
            self.dataset.allocation_manifest_path is not None
            or self.training.portability_manifest_path is not None
        ):
            raise ValueError(
                "compiled output head and loss do not support allocation "
                "or portability runs"
            )
        if self.tokenizer.origin is not None and self.dataset.source not in {
            "snapshot",
            "local_text",
            "local_chat",
        }:
            raise ValueError(
                "explicit tokenizer origin requires a content-bound consumer"
            )
        if (
            self.model.memory_conv_kernel_size
            and self.dataset.allocation_manifest_path is not None
        ):
            raise ValueError("convolved token memory does not support allocation masks")
        if self.evaluation.panels:
            if (
                self.runtime.engine != "pytorch"
                or self.model.memory not in {"none", "ngram"}
                or self.model.ffn != "dense"
                or self.dataset.allocation_manifest_path is not None
                or self.model.semantic_memory_dim is not None
            ):
                raise ValueError(
                    "fixed panels require dense-FFN PyTorch with no memory or token ngram memory"
                )
            if any(
                p.sequence_length != self.training.seq_len
                for p in self.evaluation.panels
            ):
                raise ValueError(
                    "fixed panel sequence length must match training.seq_len"
                )
        if self.training.seq_len > self.model.max_seq_len:
            raise ValueError("training.seq_len cannot exceed model.max_seq_len")
        if self.training.portability_manifest_path is not None:
            if self.runtime.engine != "pytorch":
                raise ValueError(
                    "portability runs require the PyTorch engine in this implementation"
                )
            if self.training.trainable_parameters is None:
                raise ValueError(
                    "portability runs require an explicit trainable_parameters list"
                )
        if (
            self.attention.latent_dim is not None
            and self.attention.latent_dim % self.model.num_heads
        ):
            raise ValueError("attention.latent_dim must divide evenly across heads")
        if (
            self.model.num_kv_heads is not None
            and self.model.num_kv_heads != self.model.num_heads
            and (
                self.attention.kind in {"mla", "block_sparse", "local_selected"}
                or self.runtime.engine == "mlx"
            )
        ):
            raise ValueError(
                "grouped-query attention is supported only for PyTorch dense or sliding_window attention"
            )
        if self.attention.implementation == "sdpa" and (
            self.runtime.engine != "pytorch"
            or (
                self.model.num_kv_heads is not None
                and self.model.num_kv_heads != self.model.num_heads
            )
        ):
            raise ValueError(
                "SDPA requires PyTorch dense attention with equal Q/KV head counts"
            )
        policy = self.checkpoint.retention
        if policy is not None and (
            policy.terminal_step != self.training.max_steps
            or policy.terminal_targets != self.training.max_tokens
        ):
            raise ValueError("retention terminal boundary must match training limits")
        if self.optimizer.warmup_steps >= self.training.max_steps:
            raise ValueError(
                "optimizer.warmup_steps must be smaller than training.max_steps"
            )
        if (
            isinstance(self.optimizer, AdamWConfig)
            and self.optimizer.decay_steps is not None
        ):
            if not (
                self.optimizer.warmup_steps
                < self.optimizer.decay_steps
                <= self.training.max_steps
            ):
                raise ValueError(
                    "optimizer.decay_steps must exceed warmup_steps and not exceed training.max_steps"
                )
            if self.runtime.engine == "mlx":
                raise ValueError("optimizer.decay_steps is unsupported for MLX")
        if self.training.stable_steps:
            if (
                not isinstance(self.optimizer, AdamWConfig)
                or self.runtime.engine != "pytorch"
            ):
                raise ValueError("stable_steps requires PyTorch AdamW")
            horizon = self.optimizer.decay_steps or self.training.max_steps
            if self.optimizer.warmup_steps + self.training.stable_steps >= horizon:
                raise ValueError(
                    "warmup_steps + stable_steps must be below decay horizon"
                )
        if (
            isinstance(self.optimizer, AdamWConfig)
            and (
                self.optimizer.memory_table_lr_multiplier != 1.0
                or self.optimizer.memory_table_weight_decay is not None
            )
            and (self.runtime.engine != "pytorch" or self.model.memory != "ngram")
        ):
            raise ValueError(
                "memory table optimizer policy requires PyTorch ngram memory"
            )
        if self.optimizer.floor > self.optimizer.peak:
            raise ValueError("optimizer.floor cannot exceed optimizer.peak")
        if isinstance(self.optimizer, AdamWConfig) and any(
            not 0 <= beta < 1 for beta in self.optimizer.betas
        ):
            raise ValueError("optimizer.betas entries must be in [0, 1)")
        if self.optimizer.state_offload:
            raise ValueError("optimizer.state_offload is deferred and unsupported")
        if self.runtime.engine == "mlx" and self.runtime.backend != "metal":
            raise ValueError("MLX engine requires runtime.backend=metal")
        if self.runtime.engine == "pytorch" and self.runtime.backend == "metal":
            raise ValueError("PyTorch runtime does not use backend=metal")
        if (
            self.runtime.engine == "mlx"
            and self.model.num_kv_heads is not None
            and self.model.num_kv_heads != self.model.num_heads
        ):
            raise ValueError("MLX engine does not support grouped-query attention")
        if (
            self.model.semantic_memory_dim is not None
            and self.dataset.allocation_manifest_path is None
        ):
            raise ValueError(
                "model.semantic_memory_dim requires dataset.allocation_manifest_path"
            )
        if (
            self.dataset.allocation_manifest_path is not None
            and self.runtime.engine != "pytorch"
        ):
            raise ValueError(
                "allocation ownership currently requires the PyTorch engine"
            )
        if self.runtime.secondary_device_indices and (
            self.model.semantic_memory_dim is not None
            or self.dataset.allocation_manifest_path is not None
            or self.training.portability_manifest_path is not None
        ):
            raise ValueError(
                "runtime.secondary_device_indices cannot be combined with semantic memory, "
                "allocation sidecars, or portability runs"
            )
        return self


class TokenizerSampling(StrictModel):
    """Deterministic whole-document byte quotas over an authenticated TRAIN view."""

    source_strata: dict[str, str]
    byte_weights: dict[str, Annotated[int, Field(strict=True, gt=0)]]
    max_fit_bytes: int = Field(strict=True, gt=0, le=512 * 1024 * 1024)
    audit_documents_per_stratum: int = Field(default=1000, strict=True, gt=0, le=10000)
    max_audit_bytes: int = Field(
        default=64 * 1024 * 1024, strict=True, gt=0, le=512 * 1024 * 1024
    )

    @model_validator(mode="after")
    def strata(self):
        if (
            not self.byte_weights
            or set(self.source_strata.values()) != set(self.byte_weights)
            or any(
                not value.strip() for value in (*self.source_strata, *self.byte_weights)
            )
            or sum(self.byte_weights.values()) > 10000
        ):
            raise ValueError(
                "sampling requires matching source strata and bounded positive byte weights"
            )
        return self


class TokenizerTrainConfig(StrictModel):
    schema_version: Literal[1]
    vocab_size: int = Field(ge=260)
    min_frequency: int = Field(default=2, gt=0)
    max_documents: int = Field(gt=0)
    output_dir: Path
    dataset: DatasetConfig
    sampling: TokenizerSampling | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def authenticated_sampling(self):
        if self.sampling is not None and (
            self.dataset.source != "local_text"
            or self.dataset.corpus_release_path is None
            or self.dataset.corpus_export_path is None
            or self.sampling.max_fit_bytes > self.dataset.train_max_tokens
        ):
            raise ValueError(
                "stratified sampling requires a verified corpus LM export and a compatible byte budget"
            )
        return self
