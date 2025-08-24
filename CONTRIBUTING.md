# Contributing to infermesh

🎉 First of all, thank you for your interest in contributing to **infermesh**!  
We welcome contributions of all kinds: code, documentation, design proposals, and community support.

---

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).  
Please read it to understand the expectations for all contributors.

---

## How to Contribute

### 1. Reporting Issues
- Use the [GitHub Issues](https://github.com/redbco/infermesh/issues) tracker.
- Provide clear reproduction steps, logs, or metrics where applicable.
- Label issues appropriately: `bug`, `enhancement`, `question`, `documentation`.

### 2. Proposing Features
- Open an issue labeled `proposal` or `enhancement`.
- Provide context: what problem are you solving? why does it belong in infermesh?
- Discuss design options before implementation.

### 3. Submitting Code
1. Fork the repo and create your branch from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```
2. Run the lints and tests locally:
   ```bash
   cargo fmt --all -- --check
   cargo clippy --all-targets --all-features -- -D warnings
   cargo nextest run --all
   ```
3. Commit with clear messages (see **Commit Guidelines** below).
4. Push to your fork and open a Pull Request (PR).

### 4. Improving Documentation
- Fix typos, clarify explanations, or add examples.  
- Documentation-only PRs are always welcome.

### 5. Contributing Dashboards & Configs
- Add example PromQL queries, Grafana dashboards, or Kubernetes manifests under `docs/` or `examples/`.

---

## Development Setup

### Prerequisites
- Rust (stable, via rustup)
- `protoc` (protobuf compiler)
- Docker (optional, for container builds)
- NVIDIA drivers + CUDA (for GPU nodes)

### Build All Crates
```bash
cargo build --workspace --release
```

### Run Tests
```bash
cargo nextest run --all
```

### Linting
```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

---

## Commit Guidelines

We follow a **conventional commit** style:

- `feat:` new feature  
- `fix:` bug fix  
- `docs:` documentation changes  
- `style:` formatting or stylistic changes (no logic)  
- `refactor:` code restructuring without behavior change  
- `test:` adding or updating tests  
- `chore:` maintenance tasks

Example:
```
feat(router): add hedged requests for p99 tail latency
```

---

## Pull Request Checklist

- [ ] Code compiles and tests pass locally  
- [ ] Lints pass (fmt + clippy)  
- [ ] Documentation updated (README, ARCHITECTURE, etc. if applicable)  
- [ ] Added tests for new functionality  
- [ ] PR description explains what and why, not just how  

---

## Release Process

1. Maintainers update `CHANGELOG.md` with highlights.
2. Version bump in `Cargo.toml` (semver).
3. Tag a release:  
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```
4. GitHub Actions builds and publishes release artifacts (binaries, containers).

---

## Getting Help

- [GitHub Discussions](https://github.com/redbco/infermesh/discussions)
- [Issues](https://github.com/redbco/infermesh/issues)

---

Thanks again for helping improve **infermesh** 💜
