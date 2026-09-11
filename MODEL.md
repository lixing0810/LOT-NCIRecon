# Model checkpoint

The validated arterial-phase checkpoint is not committed because the full training snapshot is approximately 1.47 GiB.

Place it at:

```text
outputs/reconstruction/A/checkpoints/latest.pth
```

Validated checkpoint metadata:

- filename: `latest.pth`
- size: `1,574,076,724` bytes
- SHA-256: `658e1084f5a68e51887d28db42794b674be83a356ecf017ee154e7d57bfd5931`
- training step: `100000`
- generator tensors: `668`
- generator parameters: `17,468,197`
- EMA weights: not present
- phase: arterial (`A`)

Strict compatibility was checked against `lot_ncirecon.models.LOTNCIRecon`: 668/668 keys matched, with zero missing keys, zero unexpected keys, and zero shape mismatches. The checkpoint contains generator, discriminator, optimizer, scaler and argument state. `scripts/test.py` automatically extracts the `generator` state.

The supplied file was `latest.pth`; the originally mentioned `latest.py` filename did not exist.
