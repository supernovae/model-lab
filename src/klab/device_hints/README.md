# Device performance hints

One YAML file per device family. `klab inspect --explain` prints the hints of
the family that matches the runtime's device. A hint is a suggestion with the
measurements behind it; Lab never applies one.

To add your hardware, run `klab runtime tune CONFIG --runtime ID` on it, then
copy the `trials` rows that helped into a new `<family>.yaml` here (see
`xpu-intel-arc-pro-b70.yaml`). Keep `evidence` as measured, including trials
that did not help.
