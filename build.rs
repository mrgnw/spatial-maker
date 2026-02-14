use std::env;
use std::process::Command;

fn main() {
	#[cfg(target_os = "macos")]
	{
		if env::var("CARGO_FEATURE_COREML").is_ok() {
			println!("cargo:rerun-if-changed=swift-bridge/CoreMLDepth.swift");

			let out_dir = env::var("OUT_DIR").unwrap();
			let lib_path = format!("{}/libCoreMLDepth.a", out_dir);

			let status = Command::new("swiftc")
				.args(&[
					"-emit-library",
					"-static",
					"-module-name",
					"CoreMLDepth",
					"-O",
					"swift-bridge/CoreMLDepth.swift",
					"-o",
					&lib_path,
				])
				.status()
				.expect("Failed to compile Swift code. Is Xcode Command Line Tools installed?");

			if !status.success() {
				panic!("swiftc compilation failed");
			}

			println!("cargo:rustc-link-search=native={}", out_dir);
			println!("cargo:rustc-link-lib=static=CoreMLDepth");

			println!("cargo:rustc-link-lib=framework=CoreML");
			println!("cargo:rustc-link-lib=framework=Foundation");
			println!("cargo:rustc-link-lib=framework=CoreVideo");
			println!("cargo:rustc-link-lib=framework=Accelerate");

			let sdk_path = String::from_utf8(
				Command::new("xcrun")
					.args(&["--show-sdk-path"])
					.output()
					.expect("Failed to find SDK path")
					.stdout,
			)
			.unwrap();

			let sdk_path = sdk_path.trim();
			println!("cargo:rustc-link-search=native={}/usr/lib/swift", sdk_path);
			println!("cargo:rustc-link-search=native=/usr/lib/swift");
		}
	}
}
