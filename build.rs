use std::env;
use std::process::Command;

fn main() {
    // Only build Swift bridge on macOS
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rerun-if-changed=swift-bridge/CoreMLDepth.swift");

        let out_dir = env::var("OUT_DIR").unwrap();
        let lib_path = format!("{}/libCoreMLDepth.a", out_dir);

        // Compile Swift to static library
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

        // Link Apple frameworks
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=CoreVideo");
        println!("cargo:rustc-link-lib=framework=Accelerate");

        // Link Swift runtime
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
