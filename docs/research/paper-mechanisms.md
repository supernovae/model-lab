# Paper mechanisms and extension requirements

Detailed reading notes supporting [Papers and experiments](../../papers.md).
Entries describe what a paper reports, a possible controlled lab test, and its
limits. This is a reference catalog, not a ranked backlog or a claim of paper
reproduction. The runnable routes and recorded findings are indexed in the main
guide; proposals here do not authorize experiments or alter existing protocols.

Titles and primary references retain the catalog's October 2026 source review.
Pinned library documentation describes the cited version, not current backend
acceptance. New integrations need their own source and runtime checks.

## Conditional memory, transfer and adaptation

### When to Adapt: Conditional Memory Adapters for Retention-Preserving Domain Specialization

**2026 · [Primary paper](https://arxiv.org/abs/2608.29327)**

- **Contribution:** Engram Adapter adds occupancy-aware n-gram matching and gated
  residuals to a frozen model. Qwen3-4B/8B experiments report domain gains with
  strong retention on the tested out-of-domain tasks.
- **Proposed lab test:** extend the existing gated memory path with explicit
  occupancy masking; compare frozen baseline, always-on adapter and conditional
  adapter at matched adaptation budgets. Measure domain accuracy, unrelated-task
  retention, gate activity and output-distribution drift together.
- **Limit:** local matches also occur out of domain; retention depends on learned
  attenuation and the tested distributions, not a guarantee of zero interference.

### Memory Grafting: Scaling Language Model Pre-training via Offline Conditional Memory

**2026 · [Primary paper](https://arxiv.org/abs/2605.20948)**

- **Contribution:** an offline donor supplies hidden vectors for frequent n-grams;
  longest-suffix lookup, projections and gates feed a recipient, with hashed
  Engram fallback for misses. The paper reports gains over matched pretraining
  baselines.
- **Proposed lab test:** use the [compiled-initialization bridge](compiled-engram-initialization-v1.md)
  as a design reference for random, compiled-frozen and compiled-refined arms.
  A donor-vector compiler and exact suffix path would be new work. Include
  shuffled-value controls and charge donor compilation separately.
- **Limit:** recipients still train for 50B/100B tokens in the main settings.
  This is not demonstrated cheap retrofitting of an arbitrary frozen model.

### Tokenizer-Agnostic Engram Module

**2026 · [Primary paper](https://arxiv.org/abs/2607.29065)**

- **Contribution:** polynomial hashing over bytes and a shared embedding space
  across n-gram orders make byte-equivalent sequences address the same memory
  despite different tokenizations; experiments train recipients using pretrained
  Engram embeddings.
- **Proposed lab test:** build a separate cross-tokenizer study from the
  [byte/portable interfaces](../portable-engram.md), comparing transferred,
  random and permuted tables with matched recipient training. Check byte-key
  agreement separately from held-out transfer accuracy.
- **Limit:** address equivalence is not universal vector alignment. Recipient
  training remains necessary in the reported transfer setup; the existing
  [learned portability protocol](learned-engram-portability.md)
  tests widths within one model family, not this entire claim.

### Scaling Embedding Layers in Language Models

**2025 · [Primary paper](https://arxiv.org/abs/2502.01637)**

- **Contribution:** SCONE learns contextual n-gram input embeddings with a separate
  model, then precomputes and offloads them for inference without enlarging the
  output vocabulary. It explores capacity outside the decoding backbone.
- **Proposed lab test:** use embedding-site token memory as a starting comparison,
  then separately implement contextual embedding compilation. Compare ordinary
  embeddings, random tables and compiled tables at fixed backbone/token budgets;
  measure compilation cost, lookup coverage, loss and inference bytes.
- **Limit:** fixed accelerator usage during inference excludes external storage
  and the embedding model's training cost. Existing Engram lookup is not a SCONE
  reproduction or proof of its latency results.

### Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models

**2026 · [Primary paper](https://arxiv.org/abs/2601.07372)**

- **Contribution:** Engram combines hashed n-gram memory with context-dependent
  gating, reallocating sparse capacity between stored patterns and MoE compute.
  Matched large-model experiments report quality improvements.
- **Proposed lab test:** use the [Engram/MoE capacity study](engram-moe-capacity.md)
  to motivate separately budgeted memory-versus-expert ablations; report total
  parameters, active work, table utilization, held-out loss and generation quality.
- **Limit:** the under-3% host-offload throughput penalty uses host DRAM, an NVIDIA
  H800 and 512 sequences. It is not an SSD, batch-one, small-ROCm or training
  result; the lab's local reference components do not inherit that performance.

### Larimar: Large Language Models with Episodic Memory Control

**2024 · [Primary paper, v4](https://arxiv.org/html/2403.11901v4)**

- **Contribution:** a learned encoder/decoder interface couples a language model
  to an episodic memory with least-squares reads and writes. After initial
  training, new edits and selective forgetting can occur without per-edit
  gradient training; fact-editing benchmarks test speed and locality.
- **Proposed lab test:** add a distinct episodic controller beside the existing
  semantic-memory API. Compare write/read/delete behavior with retrieval-only
  and no-memory controls, including paraphrases and unrelated facts.
- **Limit:** training-free edits assume an already trained memory interface.
  Removing a memory entry does not establish that the underlying language model
  has forgotten every occurrence of that fact.

### User as Engram: Internalizing Per-User Memory as Local Parametric Edits

**2026 · [Primary paper](https://arxiv.org/abs/2606.19172)**

- **Contribution:** local hash-table edits hold individual facts while a shared
  adapter supplies reasoning skills; experiments separate direct recall from
  reasoning over inserted content and examine interference.
- **Proposed lab test:** use synthetic identities and facts in a new row-edit
  extension to byte memory. Compare table edits, an adapter baseline and
  retrieval, with collision, paraphrase, deletion and cross-identity controls.
- **Limit:** it assumes Engram pretraining and suitable injection depth. Locality
  for untouched addresses does not guarantee collision-free storage, robust
  paraphrase recall or reliable multi-hop reasoning as fact density grows.

## Attention, recurrence and model architecture

### Optimizing Native Sparse Attention with Latent Attention and Local Global Alternating Strategies

**2025 · [Primary paper](https://arxiv.org/abs/2511.00819)**

- **Contribution:** combines latent compression with local/global sparse-attention
  alternation across layers, exploring a cheaper attention/cache design in
  340M–1.3B models.
- **Proposed lab test:** extend the current uniform attention configuration with
  explicit layer schedules; compare alternating and fixed schedules at matched
  parameters, context and token budgets. Measure retrieval, loss and cache bytes.
- **Limit:** reported training uses 15B/100B tokens. Compression and reduced cache
  size do not establish a small-model ROCm latency win, and the lab's combined
  MLA/MoE/memory smoke does not implement this alternating architecture.

### DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

**2024 · [Primary paper](https://arxiv.org/abs/2405.04434)**

- **Contribution:** MLA compresses attention's key/value representation, alongside
  sparse DeepSeekMoE computation, to reduce inference costs at large scale.
- **Proposed lab test:** isolate latent attention against dense and grouped-query
  controls using the lab's reference paths. Compare quality, cache growth and
  decode timing; implementing the paper's compressed-cache design is distinct
  from the current [expanded-key cache](../../src/klab/model/attention/latent.py).
- **Limit:** the 236B-total/21B-active model trains on 8.1T tokens and combines
  several changes. Its headline savings are not predictions for the simplified
  lab MLA implementation or tiny-model quality.

### Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention

**2026 · [Primary paper](https://arxiv.org/abs/2605.22791)**

- **Contribution:** separates channel-wise erase and write gates in recurrent
  memory, with channel-wise decay, to improve selective state updates; evaluates
  a 1.3B model trained on 100B tokens.
- **Proposed lab test:** add a new recurrent mixer, then compare tied and separate
  gates on repeated-key replacement, interference and retrieval at equal state
  size and training budget, followed by the existing dense language baseline.
- **Limit:** bounded recurrent state still compresses history. Better retrieval
  on the reported tasks implies neither exact arbitrary recall nor efficient
  kernels on the lab's target runtime.

### On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability

**2026 · [Primary paper](https://arxiv.org/abs/2608.30320)**

- **Contribution:** studies hybrid Gated DeltaNet/QSA mixing, widened gated
  residual streams, host-prefetched n-gram embeddings and training choices as
  interacting architecture components.
- **Proposed lab test:** begin with an isolated memory-capacity sweep using the
  lab's embedding memory; compare held-out task scores as well as loss and total
  storage. Recurrent mixing, residual changes and host prefetch require separate
  extensions and ablations.
- **Limit:** downstream gains can saturate while loss improves. The
  [official Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
  lists 125B main parameters, 51B n-gram embeddings and 4B MTP, with 6B active:
  this is not a sub-35B-total model or a small-device performance demonstration.
  The [Qwen-specific vLLM recipe](https://github.com/vllm-project/recipes/blob/8faeba99ff3445c151d4758abb06060388cba9f3/models/Qwen/Qwen3.8-Flash-Next.yaml)
  documents its `VLLM_PLE_CPU_OFFLOAD` host-memory path as NVIDIA-only;
  general UVA/CUDA-alike capability does not establish this path's ROCm support.
  [llama.cpp lazy reads](https://github.com/ggml-org/llama.cpp/blob/c06f84160a30c66d7b5a2829ae9b3ea15275cbc3/tools/cli/README.md)
  are a separate mmap-based disk path. Neither establishes Kybern Model Lab ROCm
  support or performance.

### Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

**2025 · [Primary paper](https://arxiv.org/abs/2502.11089)**

- **Contribution:** combines compressed context, selected token blocks and a
  local window in end-to-end trainable, hardware-aware sparse attention.
- **Proposed lab test:** use [block-sparse attention](../sparse-attention.md) as
  a reference starting point, then add explicit three-path ablations against
  dense attention. Sweep context length and selection budget; report quality,
  selected-key counts, end-to-end time and peak memory.
- **Limit:** the paper's 27B-model/260B-token experiment and A100 kernel timings
  do not validate the lab's simpler selector or predict benefits at short context
  on ROCm. Kernel correctness and model-quality equivalence are separate gates.

### DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models

**2025 · [Primary paper](https://arxiv.org/abs/2512.02556)**

- **Contribution:** DeepSeek Sparse Attention (DSA) uses a lightweight learned
  indexer to select historical tokens for MLA, reducing long-context attention
  work; the report also includes broader training and reasoning advances.
- **Proposed lab test:** add an indexer as a new selector and compare it with
  fixed-window, random and current block selection at equal retained-token
  budgets. Count indexer overhead, missed relevant tokens and downstream quality.
- **Limit:** DSA adaptation includes 2.1B indexer warm-up tokens and 943.7B sparse
  continued-pretraining tokens. This does not demonstrate a cheap retrofit or
  isolate all full-model gains as effects of sparse attention.

### Mamba-3: Improved Sequence Modeling using State Space Principles

**2026 · [Primary paper](https://arxiv.org/abs/2603.15569)**

- **Contribution:** improves state-space sequence modeling through discretization,
  complex-valued state updates and a multi-input/multi-output formulation,
  targeting state tracking and inference efficiency.
- **Proposed lab test:** implement a separate state-space mixer; compare controlled
  state tracking and associative retrieval with recurrent and dense baselines,
  then measure language loss under matched training and state-memory budgets.
- **Limit:** the reported 1.5B-scale quality and specialized inference results
  do not establish a drop-in replacement for the current decoder attention or
  comparable speed on small ROCm workloads.

## Tool use, constrained decoding and structured reasoning

### ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings

**2023 · [Primary paper](https://arxiv.org/abs/2305.11554)**

- **Contribution:** learns output embeddings for tool tokens while freezing the
  language model; selecting a token invokes argument generation and tool use.
- **Proposed lab test:** extend the tokenizer/output head and add a bounded mock
  executor around [instruction training](../instruction-training.md). Compare
  learned tool tokens with textual descriptions under identical demonstrations;
  score selection, arguments and execution separately.
- **Limit:** experiments use LLaMA-13B/33B, not proof for 1–2B models. Embeddings
  still need training examples, and correct selection alone does not establish
  useful execution. The lab currently has no ToolkenGPT execution pipeline.

### PICARD: Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models

**2021 · [Primary paper](https://arxiv.org/abs/2109.05093)**

- **Contribution:** incremental parsing rejects inadmissible next tokens during
  decoding; text-to-SQL experiments use fine-tuned T5 models.
- **Proposed lab test:** add a parser-backed decoding constraint for a tiny tool
  grammar or structured output. Compare constrained and unconstrained generation
  from the same checkpoint, recording syntax validity, task correctness and
  decoding latency independently.
- **Limit:** a valid parse cannot ensure the right query or argument meaning.
  A parser and tokenizer-aware candidate checks are new work; this is not an
  existing guarantee of the lab's generation API.

### Octopus v2: On-device language model for super agent

**2024 · [Primary paper, v7](https://arxiv.org/html/2404.01744v7)**

- **Contribution:** dedicated functional tokens represent APIs in a fine-tuned
  Gemma-based 2B model, reducing repeated tool-description context; the paper
  explores full and LoRA training. The linked revision is from September 2026.
- **Proposed lab test:** add a fixed synthetic tool inventory and compare function
  tokens, ordinary names and descriptions. Keep examples and tool coverage
  matched; measure argument accuracy and task success alongside prompt savings.
- **Limit:** specialized function-calling results do not establish general agent
  reliability or zero-shot support for arbitrary new tools. New inventories need
  training coverage, and fewer prompt tokens alone do not prove faster execution.

### ReaRev: Adaptive Reasoning for Question Answering over Knowledge Graphs

**2022 · [Primary paper](https://arxiv.org/abs/2210.13650)**

- **Contribution:** graph feedback revises question-derived instructions while a
  graph neural network adaptively carries out multi-step reasoning.
- **Proposed lab test:** build a separate structured-memory benchmark beside the
  semantic retrieval interface. Compare adaptive revision with fixed instruction
  order and direct graph lookup on known paths, missing edges and distractors.
- **Limit:** ReaRev assumes graph structure and relevant entities; it is not a
  demonstrated drop-in decoder memory module. The lab's exact vector retrieval
  does not itself supply a graph, entity linker or learned reasoning controller.

## Inference storage and offload

### Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models

**2024 · [Primary paper](https://arxiv.org/abs/2402.07033)**

- **Contribution:** orchestrates CPU expert execution versus expert-weight transfer
  to GPU, using expert popularity to manage residency during MoE inference.
- **Proposed lab test:** extend local Top-K MoE with a separate inference-only
  placement experiment. Compare fully resident, CPU-executed and transferred
  experts; measure prefill/decode time, transferred bytes and numerical parity.
- **Limit:** gains depend on expert size, token load, CPU kernels and interconnect.
  The current local MoE keeps its weights resident; neither its sparsity nor
  Fiddler's results prove ROCm speedups or feasible gradient/optimizer offload.

### LLM in a flash: Efficient Large Language Model Inference with Limited Memory

**2023 · [Primary paper](https://arxiv.org/abs/2312.11514)**

- **Contribution:** selectively loads predicted active feed-forward weights from
  flash, reuses recently active neurons and bundles rows/columns to improve I/O.
- **Proposed lab test:** add an inference-only weight loader and compare naive,
  cached and bundled reads at equal memory budgets. Track actual bytes read,
  predictor misses, latency and quality relative to fully resident execution.
- **Limit:** useful activation sparsity and accurate prediction are prerequisites;
  the lab's dense SwiGLU checkpoints do not automatically qualify. This concerns
  inference weight loading, not training-state offload or disk checkpoint writes.

### KVSwap: Disk-aware KV Cache Offloading for Long-Context On-device Inference

**2025 · [Primary paper](https://arxiv.org/abs/2511.11907)**

- **Contribution:** keeps the full KV cache on disk, using compact metadata to
  predict preloads and schedule reads around computation and storage behavior.
- **Proposed lab test:** extend the lab's request-local cache with an explicit
  disk-backed inference mode. Compare resident and offloaded caches on identical
  long prompts, recording memory, I/O, decode latency and generation quality.
- **Limit:** the main evaluation uses Jetson Orin with NVMe/eMMC. Cache offload
  leaves model weights and training state as separate costs, and its storage
  behavior requires new measurements on discrete ROCm systems.

## Optimizer memory

These are independent research directions. AdamW and Adafactor have native lab
implementations; other optimizer and training-system entries require integration.
Published results do not establish local quality, speed or memory fit.

### Fit and comparison boundaries

The current [PyTorch optimizer factory](../../src/klab/training/optimizer.py)
provides AdamW and Adafactor with `foreach=False`; other choices below need new
integration. AdamW separates matrix and vector decay groups and deduplicates tied
parameters. Adafactor uses one decay group and its own parameter-scaling
semantics. The
[configuration](../../src/klab/config/models.py) rejects optimizer-state offload;
runs use one process/device, and the existing [activation offload](../offload.md)
does not move optimizer state.

[Memory accounting](../memory.md) keeps FP32 parameters and gradients under
autocast. Two FP32 AdamW moments alone cost about 8 bytes per trainable parameter:
552 MB at 69M, 8 GB at 1B and 16 GB at 2B (decimal units). These exclude weights,
gradients, activations, temporary workspaces and allocator headroom. Count actual
state dtypes, quantization metadata, projection matrices, uncompressed parameter
groups and any additional master copies; do not count the same FP32 weights twice.
An optimizer-state percentage is not a total peak-memory saving or proof of fit.

A possible comparison would retain AdamW as a control, match data, tokenizer,
architecture, effective batch, sequence length, token budget and seeds, and
declare each method's tuning budget and schedule semantics. Report held-out loss
and task quality alongside measured peak allocated/reserved memory, host RSS,
steady-state update time, end-to-end time and checkpoint cost. Sparse model
routing does not imply sparse-layout gradients or eliminate resident states.

[Full resume](../checkpointing.md#full-resume) binds optimizer configuration.
The [native verifier](../../src/klab/training/checkpoints.py) currently validates
AdamW/Adafactor moment inventories, shapes and FP32 dtypes. A new optimizer needs
configuration, accounting and safe state-codec/verification work, including
quantization scales, projection matrices or reproducible seeds, refresh phase,
counters and auxiliary groups where relevant. Compare uninterrupted and resumed
updates before a scientific run. Switching optimizers is a separately declared
experiment or compatible weights-only promotion, not ordinary resume.

### 8-bit Optimizers via Block-wise Quantization

**2021 · [Paper](https://arxiv.org/abs/2110.02861) · [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)**

- **Idea/evidence:** block-wise quantization compresses Adam-family moments;
  approximately 75% less storage for quantized moments than two FP32 moments,
  before metadata and exemptions. This has a maintained implementation.
- **Possible lab experiment:** compare AdamW8bit with the existing AdamW under
  matched training settings, including checkpoint parity and actual state bytes.
- **Limits/fit:** [v0.50.2 installation docs](https://huggingface.co/docs/bitsandbytes/v0.50.2/en/installation)
  document consumer RDNA support and gfx1100 ROCm wheels. Validate the exact
  Python/PyTorch/ROCm/OS combination and numerical behavior locally. Small tensors
  below 4096 elements remain FP32 by default; weights and gradients are separate.

### Adafactor: Adaptive Learning Rates with Sublinear Memory Cost

**2018 · [Paper](https://arxiv.org/abs/1804.04235)**

- **Idea/evidence:** factor matrix second moments into row/column statistics;
  the minimal formulation omits a full first moment. An established method
  already available through the lab's native PyTorch Adafactor path.
- **Possible lab experiment:** use the existing implementation as a low-state
  comparator, measuring quality and update magnitude as well as factor bytes.
- **Limits/fit:** vectors retain unfactored variance, and tensor shape determines
  savings. Relative step-size caps and parameter-RMS scaling are not AdamW's
  learning-rate semantics; library defaults differ. The existing tiny-model
  [state accounting check](../memory.md#adafactor-state-accounting) is not a
  quality result or a total-memory benchmark.

### Adam-mini: Use Fewer Learning Rates To Gain More

**2024 · [Paper](https://arxiv.org/abs/2406.16793) · [Implementation](https://github.com/zyushun/Adam-mini)**

- **Idea/evidence:** retains full momentum but shares second-moment statistics
  within parameter blocks, approaching half Adam's moment storage. The paper
  evaluates models from 39M to 13B parameters.
- **Possible lab experiment:** compare blockwise statistics with AdamW and
  Adafactor while auditing group coverage and tied embeddings.
- **Limits/fit:** the official implementation's parameter-name rules and
  attention metadata require mapping to this model. Full momentum remains;
  approximate state savings do not imply half total memory. Author-recommended
  AdamW hyperparameters are a starting hypothesis, not a quality guarantee.

### GaLore and GaLore 2

**2024/2025 · [GaLore](https://arxiv.org/abs/2403.03507) · [GaLore 2](https://arxiv.org/abs/2504.20437)**

- **Idea/evidence:** keep optimizer statistics in refreshed low-rank gradient
  subspaces while training full weights. GaLore 2 uses randomized SVD to reduce
  projection overhead and reports a 7B/500B-token validation.
- **Possible lab experiment:** vary rank and refresh interval for selected large
  matrices, recording quality, projection peaks and update-time spikes.
- **Limits/fit:** include cached bases, SVD workspaces and unprojected groups.
  The original 24 GB headline combines low-rank states with 8-bit optimization
  and layerwise updates; plain GaLore does not inherit that fit. Per-layer
  update hooks need separate compatibility work with accumulation and clipping.
  Sparse-layout gradients are not supported by the referenced optimizer family.

### APOLLO: SGD-like Memory, AdamW-level Performance

**2024 · [Paper](https://arxiv.org/abs/2412.05270) · [Implementation](https://github.com/zhuhanqing/APOLLO)**

- **Idea/evidence:** uses random low-rank statistics to estimate channel/tensor
  scaling, then scales the full gradient. APOLLO-Mini uses rank-one, tensor-wide
  scaling; this is different from GaLore's projected update.
- **Possible lab experiment:** compare APOLLO and Mini with AdamW across seeds,
  exposing rank, scale and refresh choices as declared variables.
- **Limits/fit:** paper accounting can exclude regenerated projections, while the
  released projector caches a matrix: measure actual resident state. Preserve
  projection/refresh history in checkpoints and check dense-gradient assumptions.
  The official repository's predominantly CC-BY-NC licensing also needs review
  before incorporating code. Published gains do not establish ROCm performance.

### Symbolic Discovery of Optimization Algorithms (Lion)

**2023 · [Paper](https://arxiv.org/abs/2302.06675) · [Implementation](https://github.com/google/automl/tree/master/lion)**

- **Idea/evidence:** sign-based updates retain one momentum buffer, half the
  moment bytes of same-dtype AdamW. The paper studies vision and language tasks.
- **Possible lab experiment:** compare language-model quality at matched token
  budgets with a separately declared learning-rate/decay search.
- **Limits/fit:** this changes the update rule, not just state precision. Authors
  suggest learning rates 3–10 times smaller and decay 3–10 times larger than
  AdamW; copying AdamW settings is not an equivalent comparison.

### Muon is Scalable for LLM Training

**2025 · [Paper](https://arxiv.org/abs/2502.16982) · [Implementation](https://github.com/KellerJordan/Muon)**

- **Idea/evidence:** orthogonalizes momentum updates for hidden weight matrices;
  language-model studies motivate a quality/efficiency comparison. A
  [native PyTorch implementation](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Muon.html)
  is available for 2D parameters.
- **Possible lab experiment:** compare Muon for eligible hidden matrices plus
  auxiliary AdamW for embeddings/output and other parameters against all-AdamW.
- **Limits/fit:** group boundaries and tied weights matter. Count the auxiliary
  optimizer and Newton–Schulz temporaries as well as momentum; one buffer does
  not mean half total memory. Standard PyTorch operations make ROCm evaluation
  plausible, but do not establish speed or stability on the selected device.

### Memory-Efficient LLM Pretraining via Minimalist Optimizer Design (SCALE)

**2025 · [Paper](https://arxiv.org/abs/2506.16659) · [ICML 2026 proceedings](https://proceedings.mlr.press/v306/glentis26a.html) · [Implementation](https://github.com/OptimAI-Lab/Minimalist_LLM_Pretraining)**

- **Idea/evidence:** normalizes gradient columns, retaining momentum for the
  output head and AdamW for 1D parameters. A newer, experimental way to reduce
  hidden-matrix optimizer state.
- **Possible lab experiment:** compare layer/group choices and quality against
  AdamW, with explicit handling of the lab's tied input/output embeddings.
- **Limits/fit:** the quoted use of 35–45% of baseline memory models BF16 weights plus
  optimizer storage, excluding gradients, activations and workspaces. Main runs use
  sequence length 256; neither accounting nor results prove longer-context fit
  under the lab's FP32 parameter policy.

### COAT: Compressing Optimizer States and Activation for Memory-Efficient FP8 Training

**2024 · [Paper](https://arxiv.org/abs/2410.19313) · [Implementation](https://github.com/NVlabs/COAT)**

- **Idea/evidence:** combines FP8 optimizer-state compression with activation
  quantization. The reported 1.54× end-to-end memory reduction is about 35% less
  memory in the tested setup, not an optimizer-only saving.
- **Possible lab experiment:** isolate state compression from activation changes
  if a compatible implementation becomes available; compare numerical stability.
- **Limits/fit:** the official release uses CUDA/`qoptim_cuda` and demonstrates
  H100 execution. This is a research reference, not a verified Radeon drop-in;
  support for an FP8 dtype alone does not supply the required kernels.

### ZeRO, ZeRO-Offload and ZeRO-Infinity

**2019/2021 · [ZeRO](https://arxiv.org/abs/1910.02054) · [ZeRO-Offload](https://arxiv.org/abs/2101.06840) · [ZeRO-Infinity](https://arxiv.org/abs/2104.07857)**

- **Idea/evidence:** ZeRO stages partition optimizer states, gradients and
  parameters across workers. Offload moves states/compute to CPU; Infinity adds
  heterogeneous CPU/NVMe storage. These are training-system approaches that can
  retain AdamW, rather than competing update rules.
- **Possible lab experiment:** a separately integrated CPU-state-offload arm
  could measure device capacity versus host RAM, transfer time and update latency.
- **Limits/fit:** ordinary sharding offers no world-size capacity benefit on one
  GPU. CPU/NVMe offload requires bandwidth and substantial host/storage capacity;
  it is unrelated to saving disk checkpoints. [DeepSpeed's documentation](https://deepspeed.readthedocs.io/en/latest/zero3.html)
  describes these mechanisms, but general AMD support does not verify CPUAdam
  build compatibility or performance on a separately provisioned runtime.
