# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]
### Added
- Initial project setup with Rust workspace
- Core crates: `mesh-core`, `mesh-proto`, `mesh-agent`, `mesh-router`
- Protobuf definitions for control, state, scoring APIs
- Basic gossip membership and state fusion (mock mode)
- Prometheus metrics endpoints for agent and router
- OpenTelemetry tracing support (optional)

### Changed
- N/A

### Fixed
- N/A

### Removed
- N/A

---

## [0.1.0] - YYYY-MM-DD
### Added
- First tagged release
- Mock runtime + GPU adapters
- Single-node end-to-end demo
- Basic CLI for node discovery and scoring requests

---

[Unreleased]: https://github.com/redbco/infermesh/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/redbco/infermesh/releases/tag/v0.1.0
