# Post-training: instruct, distill, prefer, export

Three stages and an export on a laptop CPU, in about a minute. The
[post-training guide](../../docs/post-training.md) explains each setting.

The data is two kinds of situation. A clear one has a known fix ("The printer
is unplugged"). A complex one has no agreed cause ("The team is split over the
new plan"). The instruction data answers both kinds either way, at random: a
direct action or a probing question. The preference pairs then say which reply
fits which situation. `make_data.py` regenerates all four files.

Copy the directory out of the checkout so `work/` is created outside Git:

```sh
cp -r examples/post-training "$KLAB_WORK_DIR/post-training"
cd "$KLAB_WORK_DIR/post-training"

klab tokenizer train tokenizer.yaml                  # ChatML markers become tokens
klab train 1-instruct.yaml --run-id instruct         # next-token loss on replies
klab train 2-distill.yaml --run-id distill           # smaller model, instruct as teacher
klab train 3-prefer.yaml --run-id prefer             # DPO from instruct as parent

klab chat instruct --runs-dir work/runs --max-new-tokens 24 \
  --message "The scooter has a loose cable. What should I do?"
klab chat prefer --runs-dir work/runs --max-new-tokens 24 \
  --message "The scooter has a loose cable. What should I do?"

klab weights export work/runs/prefer work/export/prefer
```

In one run of these commands, `instruct` answered the scooter question with a
probing question and `prefer` answered "Tighten the cable."; a complex
situation kept its probing reply in both.

| Stage | Config | Starts from | Objective | Data |
| --- | --- | --- | --- | --- |
| instruct | `1-instruct.yaml` | fresh weights | `next_token` | `data/chat-*.jsonl` |
| distill | `2-distill.yaml` | fresh weights, a fifth of the parameters | `distill`, teacher `instruct` | `data/chat-*.jsonl` |
| prefer | `3-prefer.yaml` | `instruct` | `preference`, reference `instruct` | `data/pairs-*.jsonl` |

To compare two preference settings from the same parent in lab mode:

```sh
cat > gentler.yaml <<'YAML'
question: Does a tighter hold on the parent keep more of the chosen replies' likelihood?
set:
  training.objective.params.beta: 3.0
YAML
klab try gentler.yaml --vs 3-prefer.yaml
```

The data is 160 training and 32 held-out situations built from a small word
grid. It proves the path works and says nothing about model quality.
