# Gradient bottleneck

The output head maps a hidden state to the vocabulary, so the network behind it
receives only the part of the logit gradient that lies in the head's column
space: at most the head's rank in directions, out of the vocabulary's.

At every evaluation, Lab measures this on the held-out targets the validation
loss uses:

- **kernel_norm_fraction**: the share of the gradient's norm lost in the head.
  Near 1 means almost nothing useful passes back.
- **kept_cosine**: how closely the kept part points along the full gradient.
- **kept_squared_fraction**: the kept cosine squared. A gradient in a random
  direction would keep head rank divided by vocabulary size.

Read how these move beside the loss: a step in the loss with a step here points
at the head; a loss step with a flat line here points elsewhere. See
[Evaluation](https://github.com/supernovae/model-lab/blob/main/docs/evaluation.md#gradient-bottleneck).
