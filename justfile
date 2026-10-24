set shell := ["bash", "-euo", "pipefail", "-c"]

version := `grep '^version' Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/'`
tag := "v" + version

# Build (debug)
build:
	cargo build

# Build (release)
build-release:
	cargo build --release

# Build with temporary target dir, then clean that target on success
build-release-clean:
	DEVCLEAN_RUST_TARGET_BASE="${DEVCLEAN_RUST_TARGET_BASE:-/Volumes/wd/cache/rust-targets}" ../scripts/rust-build-clean . --release

# Build release archives for macOS targets
# Uses a temporary target dir and removes it after successful packaging.
dist:
	#!/bin/bash
	set -euo pipefail
	bin="spatial-maker"
	dist="dist"
	targets=(
		aarch64-apple-darwin
	)
	target_base="${DEVCLEAN_RUST_TARGET_BASE:-/Volumes/wd/cache/rust-targets}"
	target_dir="${target_base}/spatial-maker-dist"
	echo "building ${bin} {{tag}}"
	echo
	rm -rf "${dist}"
	mkdir -p "${dist}"
	mkdir -p "${target_base}"
	for target in "${targets[@]}"; do
		echo "--- ${target}"
		CARGO_TARGET_DIR="${target_dir}" cargo build --release --target "${target}"
		archive="${bin}-{{tag}}-${target}.tar.gz"
		tar -czf "${dist}/${archive}" -C "${target_dir}/${target}/release" "${bin}"
		echo "  -> ${dist}/${archive}"
		echo
	done
	rm -rf "${target_dir}"
	echo "done"
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
	echo "  https://github.com/mrgnw/spatial-maker/releases/tag/{{tag}}"

# Install locally
install:
	cargo install --path .

# Upload CoreML models to HuggingFace
upload-models:
	#!/bin/bash
	set -euo pipefail
	for size in Small Base Large; do
		f="checkpoints/DepthAnythingV2${size}F16.mlpackage.tar.gz"
		if [[ -f "$f" ]]; then
			echo "uploading ${f}..."
			hf upload mrgnw/depth-anything-v2-coreml "$f" "$(basename "$f")"
		else
			echo "missing ${f}, skipping"
		fi
	done
	echo "done"
