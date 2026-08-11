# Security policy

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository rather
than opening a public issue with exploit details.

## Dependency constraints

This project pins a narrow, validated model-conversion toolchain. Two upstream
security advisories cannot currently be resolved by version upgrades without
leaving the dependency range supported by `coreai-torch 0.4.1`:

- `coreai-torch 0.4.1` requires PyTorch `>=2.8.0,<=2.11.0`. The PyTorch advisory
  affecting versions through 2.12.1 is specific to `torch.jit.script`; this
  project uses `torch.export.export` and does not call or accept input for
  `torch.jit.script`.
- PyTorch 2.11.0 requires `setuptools<82`. The setuptools advisory fixed in
  83.0.0 concerns `MANIFEST.in` exclusion bypasses while building source
  distributions on normalization-sensitive macOS filesystems. This project
  builds its source distribution and wheel with Hatchling, not setuptools.

These are reviewed compatibility exceptions, not claims that the affected
packages are universally safe. They should be removed as soon as an updated
Core AI toolchain permits patched versions. Avoid running untrusted model code,
checkpoints, build trees, or Python scripts in this environment.
