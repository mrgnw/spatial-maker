import CoreML
import CoreVideo
import Foundation
import Accelerate

// MARK: - C API for Rust FFI

@_cdecl("coreml_load_model")
public func loadModel(_ pathPtr: UnsafePointer<CChar>) -> UnsafeMutableRawPointer? {
	let path = String(cString: pathPtr)
	let url = URL(fileURLWithPath: path)
	
	// Try to load the model, compiling if needed
	do {
		// Check if it's an .mlpackage that needs compilation
		let modelURL: URL
		if path.hasSuffix(".mlpackage") {
			// Compile the model first
			let compiledURL = try MLModel.compileModel(at: url)
			modelURL = compiledURL
			print("✓ Model compiled to: \(compiledURL.path)")
		} else {
			modelURL = url
		}
		
		// Configure to use all compute units (ANE + GPU + CPU)
		let config = MLModelConfiguration()
		config.computeUnits = .all
		
		let model = try MLModel(contentsOf: modelURL, configuration: config)
		print("✓ CoreML model loaded successfully")
		
		// Return retained pointer to model
		return Unmanaged.passRetained(model as AnyObject).toOpaque()
	} catch {
		print("Failed to load CoreML model at: \(path)")
		print("Error: \(error)")
		return nil
	}
}

@_cdecl("coreml_unload_model")
public func unloadModel(_ modelPtr: UnsafeMutableRawPointer) {
	Unmanaged<AnyObject>.fromOpaque(modelPtr).release()
}

@_cdecl("coreml_infer_depth")
public func inferDepth(
	_ modelPtr: UnsafeMutableRawPointer,
	_ rgbData: UnsafePointer<UInt8>,
	_ width: Int32,
	_ height: Int32,
	_ outputPtr: UnsafeMutablePointer<Float>
) -> Int32 {
	let model = Unmanaged<AnyObject>.fromOpaque(modelPtr).takeUnretainedValue() as! MLModel
	let w = Int(width)
	let h = Int(height)
	
	do {
		// Create CVPixelBuffer from RGB8 data
		// CoreML expects BGRA format, so we need to convert RGB -> BGRA
		var pixelBuffer: CVPixelBuffer?
		let bytesPerRow = w * 4
		let status = CVPixelBufferCreate(
			kCFAllocatorDefault,
			w,
			h,
			kCVPixelFormatType_32BGRA,
			nil,
			&pixelBuffer
		)
		
		guard status == kCVReturnSuccess, let pixelBuffer = pixelBuffer else {
			print("Failed to create CVPixelBuffer")
			return -5
		}
		
		// Lock pixel buffer for writing
		CVPixelBufferLockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
		defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, CVPixelBufferLockFlags(rawValue: 0)) }
		
		let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)!
		let bgraData = baseAddress.assumingMemoryBound(to: UInt8.self)
		
		// Convert RGB (HWC) to BGRA
		for i in 0..<(w * h) {
			let rgbIdx = i * 3
			let bgraIdx = i * 4
			bgraData[bgraIdx + 0] = rgbData[rgbIdx + 2]  // B
			bgraData[bgraIdx + 1] = rgbData[rgbIdx + 1]  // G
			bgraData[bgraIdx + 2] = rgbData[rgbIdx + 0]  // R
			bgraData[bgraIdx + 3] = 255                   // A
		}
		
		// Run inference with CVPixelBuffer
		let inputFeature = try MLDictionaryFeatureProvider(
			dictionary: ["image": MLFeatureValue(pixelBuffer: pixelBuffer)]
		)
		
		let result = try model.prediction(from: inputFeature)
		
		// Extract depth output
		// Output shape: [1, height, width]
		// Output type: Float16
		guard let depthArray = result.featureValue(for: "depth")?.multiArrayValue else {
			print("Failed to extract depth output")
			return -3
		}
		
		// Convert Float16 output to Float32 for Rust
		let outputCount = w * h
		let srcPtr16 = depthArray.dataPointer.assumingMemoryBound(to: Float16.self)
		for i in 0..<outputCount {
			outputPtr[i] = Float(srcPtr16[i])
		}
		
		return 0
	} catch {
		print("Inference error: \(error)")
		return -4
	}
}

@_cdecl("coreml_get_model_info")
public func getModelInfo(_ modelPtr: UnsafeMutableRawPointer) -> UnsafePointer<CChar>? {
	let model = Unmanaged<AnyObject>.fromOpaque(modelPtr).takeUnretainedValue() as! MLModel
	let desc = model.modelDescription
	let info = "Model: \(desc.metadata[.description] ?? "Unknown")"
	return UnsafePointer(strdup(info))
}
