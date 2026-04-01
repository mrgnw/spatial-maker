set shell := ["bash", "-euo", "pipefail", "-c"]

version := `grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/'`
tag := "v" + version

# Build all workspace crates (debug)
build:
	cargo build --workspace

# Build all workspace crates (release)
build-release:
	cargo build --workspace --release

# Build the UI
build-ui:
	cd ui && pnpm install && pnpm build

# Build everything: UI + all crates (release)
build-all: build-ui build-release

# Cross-compile release archives for all targets
dist: build-ui
	#!/bin/bash
	set -euo pipefail
	bin="ubermind"
	dist="dist"
	targets=(
		aarch64-apple-darwin
		x86_64-apple-darwin
		aarch64-unknown-linux-musl
		x86_64-unknown-linux-musl
	)
	echo "building ${bin} {{tag}}"
	echo
	rm -rf "${dist}"
	mkdir -p "${dist}"
	for target in "${targets[@]}"; do
		echo "--- ${target}"
		case "${target}" in
			*-apple-*)  cargo build --release --target "${target}" ;;
			*-linux-*)  cargo zigbuild --release --target "${target}" ;;
		esac
		archive="${bin}-{{tag}}-${target}.tar.gz"
		tar -czf "${dist}/${archive}" -C "target/${target}/release" "${bin}" "ubermind-daemon"
		echo "  -> ${dist}/${archive}"
		echo
	done
	echo "all builds complete"
	echo
	ls -lh "${dist}/"

# Build dist archives and create a GitHub release
release: dist
	#!/bin/bash
	set -euo pipefail
	echo
	read -p "create github release {{tag}}? [y/N] " confirm
	if [[ "${confirm}" != "y" ]]; then
		echo "skipped"
		exit 0
	fi
	gh release create "{{tag}}" \
		--title "{{tag}}" \
		--generate-notes \
		dist/*.tar.gz
	echo
	echo "released {{tag}}"
	echo "  https://github.com/mrgnw/ubermind/releases/tag/{{tag}}"
	echo
	echo "don't forget: just publish"

# Publish all crates to crates.io (in dependency order)
publish:
	cargo publish -p ubermind-core
	cargo publish -p ubermind
	cargo publish -p ubermind-daemon

# Install locally (debug build)
install:
	cargo install --path crates/ubermind-cli
	cargo install --path crates/ubermind-daemon
