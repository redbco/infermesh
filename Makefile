# Makefile for infermesh
# GPU-aware inference mesh for large-scale AI serving

.PHONY: help build test clean check fmt clippy doc install dev examples docker release

# Default target
help: ## Show this help message
	@echo "infermesh - GPU-aware inference mesh"
	@echo ""
	@echo "Available targets:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Build targets
build: ## Build all crates in release mode
	cargo build --release --workspace

build-dev: ## Build all crates in debug mode
	cargo build --workspace

build-agent: ## Build only the mesh agent
	cargo build --release -p mesh-agent

build-router: ## Build only the mesh router
	cargo build --release -p mesh-router

build-cli: ## Build only the mesh CLI
	cargo build --release -p mesh-cli

# Development targets
dev: ## Run development setup (build + check + test)
	@echo "🔧 Running development checks..."
	$(MAKE) check
	$(MAKE) test
	$(MAKE) build-dev
	@echo "✅ Development setup complete!"

check: ## Run cargo check on all crates
	cargo check --workspace

fmt: ## Format code using rustfmt
	cargo fmt --all

fmt-check: ## Check code formatting
	cargo fmt --all -- --check

clippy: ## Run clippy linter
	cargo clippy --workspace --all-targets --all-features -- -D warnings

clippy-fix: ## Run clippy with automatic fixes
	cargo clippy --workspace --all-targets --all-features --fix --allow-dirty --allow-staged

# Testing targets
test: ## Run all tests
	cargo test --workspace

test-unit: ## Run unit tests only
	cargo test --workspace --lib

test-integration: ## Run integration tests
	cargo test --workspace --test '*'

test-doc: ## Run documentation tests
	cargo test --workspace --doc

# Documentation targets
doc: ## Generate documentation
	cargo doc --workspace --no-deps --open

doc-private: ## Generate documentation including private items
	cargo doc --workspace --no-deps --document-private-items --open

# Installation targets
install: ## Install binaries to cargo bin directory
	cargo install --path crates/mesh-agent --bin meshd
	cargo install --path crates/mesh-cli --bin mesh

install-dev: ## Install binaries with debug symbols
	cargo install --path crates/mesh-agent --bin meshd --debug
	cargo install --path crates/mesh-cli --bin mesh --debug

# Example and demo targets
examples: ## Build example configurations
	@echo "📝 Generating example configurations..."
	@mkdir -p examples
	@echo "# Example single-node configuration" > examples/single-node.yaml
	@echo "node:" >> examples/single-node.yaml
	@echo "  id: \"single-node\"" >> examples/single-node.yaml
	@echo "  roles: [\"gpu\", \"router\"]" >> examples/single-node.yaml
	@echo "📝 Example configurations created in examples/"

demo: build-dev ## Run a quick demo
	@echo "🚀 Starting infermesh demo..."
	@echo "Starting mesh agent in background..."
	@cargo run -p mesh-agent -- start &
	@sleep 3
	@echo "Checking mesh status..."
	@cargo run -p mesh-cli -- list-nodes || true
	@echo "Demo complete! Stop the agent with: pkill meshd"

# Docker targets
docker: ## Build Docker images
	docker build -t infermesh/agent -f docker/Dockerfile.agent .
	docker build -t infermesh/router -f docker/Dockerfile.router .
	docker build -t infermesh/cli -f docker/Dockerfile.cli .

docker-dev: ## Build development Docker images
	docker build -t infermesh/agent:dev -f docker/Dockerfile.agent --target development .

# Maintenance targets
clean: ## Clean build artifacts
	cargo clean
	rm -rf target/
	rm -rf examples/

clean-docs: ## Clean generated documentation
	cargo clean --doc

update: ## Update dependencies
	cargo update

audit: ## Run security audit
	cargo audit

# Release targets
release-check: ## Check if ready for release
	@echo "🔍 Checking release readiness..."
	$(MAKE) fmt-check
	$(MAKE) clippy
	$(MAKE) test
	$(MAKE) doc
	@echo "✅ Release checks passed!"

release-build: ## Build release artifacts
	@echo "📦 Building release artifacts..."
	cargo build --release --workspace
	@mkdir -p dist
	@cp target/release/meshd dist/
	@cp target/release/mesh dist/
	@echo "📦 Release artifacts created in dist/"

# Utility targets
deps: ## Install development dependencies
	@echo "📦 Installing development dependencies..."
	@command -v protoc >/dev/null 2>&1 || (echo "❌ protoc not found. Please install Protocol Buffers compiler" && exit 1)
	@rustup component add rustfmt clippy
	@cargo install cargo-audit
	@echo "✅ Development dependencies installed!"

setup: deps ## Setup development environment
	@echo "🔧 Setting up development environment..."
	$(MAKE) build-dev
	$(MAKE) examples
	@echo "✅ Development environment ready!"

bench: ## Run benchmarks
	cargo bench --workspace

profile: ## Run with profiling
	cargo build --release --workspace
	@echo "🔍 Profiling build complete. Run with profiling tools as needed."

# Metrics and analysis
lines: ## Count lines of code
	@echo "📊 Lines of code:"
	@find crates -name "*.rs" -exec wc -l {} + | tail -1
	@echo "📊 Lines by crate:"
	@for crate in crates/*/; do \
		echo -n "$$(basename $$crate): "; \
		find $$crate -name "*.rs" -exec cat {} + | wc -l; \
	done

size: build ## Show binary sizes
	@echo "📊 Binary sizes:"
	@ls -lh target/release/meshd target/release/mesh 2>/dev/null || echo "Run 'make build' first"

# Git hooks
hooks: ## Install git hooks
	@echo "🪝 Installing git hooks..."
	@mkdir -p .git/hooks
	@echo '#!/bin/sh\nmake fmt-check && make clippy' > .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "✅ Git hooks installed!"

# Quick development workflow
quick: ## Quick development check (fmt + clippy + test)
	@echo "⚡ Running quick checks..."
	$(MAKE) fmt
	$(MAKE) clippy
	$(MAKE) test-unit
	@echo "✅ Quick checks passed!"

# All-in-one targets
all: ## Build everything and run all checks
	$(MAKE) clean
	$(MAKE) setup
	$(MAKE) dev
	$(MAKE) doc
	@echo "🎉 All tasks completed successfully!"

# Variables for customization
RUST_LOG ?= info
RUST_BACKTRACE ?= 1

# Export environment variables
export RUST_LOG
export RUST_BACKTRACE
