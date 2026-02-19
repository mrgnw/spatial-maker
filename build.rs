use std::env;
use std::process::Command;

fn main() {
	#[cfg(target_os = "macos")]
	{
		if env::var("CARGO_FEATURE_COREML").is_ok() {
			println!("cargo:rerun-if-changed=swift-bridge/CoreMLDepth.swift");

			let out_dir = env::var("OUT_DIR").unwrap();
			let lib_path = format!("{}/libCoreMLDepth.a", out_dir);

			let target = env::var("TARGET").unwrap();
			let swift_target = match target.as_str() {
				"aarch64-apple-darwin" => "arm64-apple-macosx13.0",
				"x86_64-apple-darwin" => "x86_64-apple-macosx13.0",
				_ => "",
			};

			let mut cmd = Command::new("swiftc");
			cmd.args(&[
				"-emit-library",
				"-static",
				"-module-name",
				"CoreMLDepth",
				"-O",
				"swift-bridge/CoreMLDepth.swift",
				"-o",
				&lib_path,
			]);
			if !swift_target.is_empty() {
				cmd.args(&["-target", swift_target]);
			}
			let status = cmd
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

			let toolchain_path = String::from_utf8(
				Command::new("xcrun")
					.args(&["--toolchain", "default", "--find", "swiftc"])
					.output()
					.expect("Failed to find Swift toolchain")
					.stdout,
			)
			.unwrap();
			if let Some(lib_dir) = std::path::Path::new(toolchain_path.trim())
				.ancestors()
				.nth(2)
				.map(|p| p.join("lib/swift/macosx"))
			{
				println!("cargo:rustc-link-search=native={}", lib_dir.display());
			}
		}
	}
}
