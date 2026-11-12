# A local API for verified checkpoints

Commands below use the [activated, provisioned environment](../README.md#first-run).

`klab serve` loads one pinned Model Lab checkpoint and exposes a small,
nonstreaming OpenAI-compatible API. It uses the same checkpoint, tokenizer,
prepared-asset and runtime validation as CLI inference. Start with
[locating and verifying a run](tinytext-model-guide.md), or the
[MODEL-0 guide](history/model-0.md). This is a local debugging interface, not a hosted
service or a claim that a base model can follow instructions.

Serving performs no dataset acquisition or preparation. Verified historical runs
remain usable with their original run-owned assets, including runs whose old
input declarations are retired for new execution. New inputs use
[`data lock` → `data snapshot` and explicit migration](datasets.md); migration
creates a new identity and is not required merely to serve a valid old run.
The server needs the validated run closure, not a bare weight file, an array
cache from `data prepare`, or a `stage --prepared-inputs` bundle by itself.

## Start one model

With `RUNS`, `RUN_ID` and `CHECKPOINT` set to a real verified run:

```sh
klab serve "$RUN_ID" \
  --runs-dir "$RUNS" --checkpoint "$CHECKPOINT" --backend cpu \
  --host 127.0.0.1 --port 8000 --model-id local-model \
  --max-new-tokens 256 --max-request-bytes 65536 \
  --request-timeout 60 --max-clients 8
```

For a provisioned accelerator use `klab` with the authorized
`--runtime ID` or `--runtime-profile PATH` as documented in
[runtime setup](runtime.md#machine-local-runtime-environments). The server
resolves lookup pointers once; changing `latest.json` does not hot-swap a running
model. Restart to select another checkpoint. The model alias is a routing name;
`GET /v1/models` exposes the pinned provenance and capabilities.

The listener accepts loopback addresses only (`127.0.0.1` or `::1`), defaults to
port 8000, and has no authentication, TLS or browser CORS integration. It is for
trusted local processes. HTTP requests cannot select local files, checkpoint
paths or remote weight URLs; `model` must match the startup alias, and unknown
fields and paths are rejected. There is no static-file route or CORS support.

Every request must send a `Host` naming the actual bound loopback IP or
`localhost`, with the actual listening port (for example, `127.0.0.1:8000` or
`localhost:8000`; IPv6 uses `[::1]:8000`). Foreign hostnames, an unbound loopback
IP and a different port are rejected. Native clients may omit `Origin`.
If supplied, `Origin` must be the exact `http://` origin for that request's
accepted Host and port; `null`, foreign origins and other ports receive HTTP 403.
For example, `Host: 127.0.0.1:8000` with `Origin: http://localhost:8000` is rejected,
even though both names refer to loopback. These checks also apply to model
listing. They do not enable a browser frontend on another origin.

It serializes generation over one loaded model, bounds
request bytes, output budget, connected clients and request duration, and uses
request-local cache state. Stop it with Ctrl-C. Cancellation is cooperative at
decode boundaries, so a currently running backend operation must return before
its resources can be released. It is not a hard GPU-kernel timeout.

## Inspect, complete, chat

```sh
curl --fail-with-body http://127.0.0.1:8000/v1/models

# Raw continuation: exactly this prefix, with no role wrapper.
curl --fail-with-body http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","prompt":"def add(a, b):","max_tokens":32,"temperature":0,"seed":42,"stream":false}'

# Plain transcript chat; choose a short output budget for early models.
curl --fail-with-body http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"What is a function?"}],"max_tokens":32,"temperature":0,"seed":42,"stream":false}'
```

Read `choices[0].text` for completion or `choices[0].message.content` for chat.
Responses include actual token usage and a `klab` extension with the exact
serialized prompt, checkpoint identity and policy. Preserve it when reporting a
surprising output; avoid logging private prompts into shared research artifacts.
A raw prompt is unchanged. Chat uses `System:`, `User:` and `Assistant:` lines
with blank-line separators, not a learned or auto-detected model template.

Each request is independent; send the whole desired conversation. Chat accepts
an optional initial system message and alternating user/assistant text messages,
ending in a user turn. The API **rejects** context overflow instead of silently
dropping history. Its full prompt plus requested output budget must fit the
native context. This differs from terminal chat, which can drop complete oldest
turns and reports the dropped count.

## Supported request contract

| Field / capability | Behavior |
| --- | --- |
| `model` | Must name the loaded alias from `/v1/models` |
| `max_tokens` | Positive bounded output budget; defaults to min(64, server cap); must fit context |
| `temperature` | Range 0–2; zero (default) is greedy; positive values sample |
| `top_k` | Model Lab integer extension, 0 through vocabulary size; zero (default) disables filtering |
| `seed` | Integer 0 through 2^64−1 (default 0); request-local RNG within the selected runtime |
| `stop` | Null, literal string or up to four strings (1–256 characters each); omitted from visible text |
| `top_p` | Only neutral `1` accepted; nucleus sampling is not implemented |
| `n`, penalties, `stream` | Only `n=1`, zero frequency/presence penalties and `stream=false` accepted |
| Streaming | `stream=true` is rejected; no fabricated streaming of completed text |
| Tools / images / audio / JSON schema / logprobs | Unsupported; requests fail explicitly |
| Unknown fields | Rejected; remove client defaults the server does not support |

EOS and stop strings produce `finish_reason: "stop"`; exhausting the output
budget produces `"length"`. Chat also stops at generated role boundaries. Token
usage counts generated model tokens, including a generated stop/EOS token when
applicable; it need not equal retokenizing the trimmed visible text. A compatible
HTTP schema says nothing about model quality or format-following ability.

If a client fails, first reproduce with the curl request above. Inspect the
structured error: wrong alias, unsupported options, malformed messages and
context overflow need request changes, not checkpoint changes. Reduce prompt
length/output budget for context errors. Slow or busy requests may need a larger
explicit timeout or fewer callers. Never relax artifact checks to fix a client
configuration problem.

## Optional Open WebUI

The CLI and Streamlit comparator are the simplest debugging routes. If you
already use Open WebUI, add an OpenAI-compatible connection to
`http://127.0.0.1:8000/v1`, refresh models and select `local-model`. Choose the
chat-completions protocol, turn response streaming **off**, and leave tools,
vision, retrieval and unsupported sampling fields off. A client UI that cannot
send `stream=false` with this narrow request contract is not supported by this
server. Consult the official
[connection guide](https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-openai-compatible/)
for your installed release's controls. Use its server-side provider connection,
not direct cross-origin browser requests: an Origin naming the WebUI's own port
will be rejected. Its outgoing Host must retain the Model Lab address and port.
This optional integration is not an
end-to-end tested Open WebUI deployment.

Open WebUI can issue extra title, tag, follow-up and search-query tasks with
additional prompts. For clean model debugging, disable those features in its
settings (documented switches include `ENABLE_TITLE_GENERATION`,
`ENABLE_FOLLOW_UP_GENERATION` and `ENABLE_TAGS_GENERATION`), or direct tasks to a
separately configured suitable model with `TASK_MODEL_EXTERNAL`. Persisted Admin
settings can override startup environment values; see the official
[configuration reference](https://docs.openwebui.com/reference/env-configuration/).
Extra tasks consume context and serialized inference time, and their structured
instructions can produce failures unrelated to your visible user prompt.

A container's `127.0.0.1` is its own network namespace. It will not reach a
host-loopback server through a normal bridge. Use an already understood local
network arrangement that shares the host loopback, or use the native CLI/client;
do not expose this unauthenticated server through a public tunnel. Instructions
or chat training, a compatible transcript format and measured behavior determine
whether a checkpoint suits a chat UI; there is no minimum parameter-count rule.
MODEL-0 is a base checkpoint with repetitive samples, so start with raw prefixes
and compare checkpoint-bound outputs before judging it as an assistant.
