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
		} else {
			modelURL = url
		}
		
		let probeConfig = MLModelConfiguration()
		probeConfig.computeUnits = .cpuOnly
		let streaming = StreamWindow.isStreaming(try MLModel(contentsOf: modelURL, configuration: probeConfig))

		// The ANE compiler stalls on the streaming model's temporal attention; GPU runs it fine.
		let config = MLModelConfiguration()
		config.computeUnits = streaming ? .cpuAndGPU : .all
		let model = try MLModel(contentsOf: modelURL, configuration: config)
		if modelURL != url {
			try? FileManager.default.removeItem(at: modelURL)
		}

		return Unmanaged.passRetained(DepthModel(model: model, stream: streaming ? StreamWindow() : nil)).toOpaque()
	} catch {
		print("Failed to load CoreML model at: \(path)")
		print("Error: \(error)")
		return nil
	}
}

@_cdecl("coreml_unload_model")
public func unloadModel(_ modelPtr: UnsafeMutableRawPointer) {
	Unmanaged<DepthModel>.fromOpaque(modelPtr).release()
}

@_cdecl("coreml_reset_stream")
public func resetStream(_ modelPtr: UnsafeMutableRawPointer) {
	Unmanaged<DepthModel>.fromOpaque(modelPtr).takeUnretainedValue().stream?.reset()
}

@_cdecl("coreml_infer_depth")
public func inferDepth(
	_ modelPtr: UnsafeMutableRawPointer,
	_ rgbData: UnsafePointer<UInt8>,
	_ width: Int32,
	_ height: Int32,
	_ outputPtr: UnsafeMutablePointer<Float>
) -> Int32 {
	let depthModel = Unmanaged<DepthModel>.fromOpaque(modelPtr).takeUnretainedValue()
	let w = Int(width)
	let h = Int(height)
	
	do {
		// Create CVPixelBuffer from RGB8 data
		// CoreML expects BGRA format, so we need to convert RGB -> BGRA
		var pixelBuffer: CVPixelBuffer?
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
		
		// Convert RGB (HWC) to BGRA, respecting CVPixelBuffer row padding
		let actualBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
		for y in 0..<h {
			for x in 0..<w {
				let rgbIdx = (y * w + x) * 3
				let bgraIdx = y * actualBytesPerRow + x * 4
				bgraData[bgraIdx + 0] = rgbData[rgbIdx + 2]  // B
				bgraData[bgraIdx + 1] = rgbData[rgbIdx + 1]  // G
				bgraData[bgraIdx + 2] = rgbData[rgbIdx + 0]  // R
				bgraData[bgraIdx + 3] = 255                   // A
			}
		}
		
		// Run inference with CVPixelBuffer
		let image = MLFeatureValue(pixelBuffer: pixelBuffer)
		let result: MLFeatureProvider
		if let stream = depthModel.stream {
			result = try stream.predict(depthModel.model, image: image)
		} else {
			result = try depthModel.model.prediction(
				from: MLDictionaryFeatureProvider(dictionary: ["image": image])
			)
		}
		
		// Extract depth output
		// Output shape: [1, height, width]
		// Output type: Float16
		guard let depthArray = result.featureValue(for: "depth")?.multiArrayValue else {
			print("Failed to extract depth output")
			return -3
		}
		
		// Convert Float16 output to Float32 for Rust, respecting MLMultiArray strides
		let srcPtr16 = depthArray.dataPointer.assumingMemoryBound(to: Float16.self)
		let strides = depthArray.strides.map { $0.intValue }
		if strides.count == 3 {
			for y in 0..<h {
				for x in 0..<w {
					let srcIdx = y * strides[1] + x * strides[2]
					outputPtr[y * w + x] = Float(srcPtr16[srcIdx])
				}
			}
		} else {
			for i in 0..<(w * h) {
				outputPtr[i] = Float(srcPtr16[i])
			}
		}
		
		return 0
	} catch {
		print("Inference error: \(error)")
		return -4
	}
}

@_cdecl("coreml_get_model_info")
public func getModelInfo(_ modelPtr: UnsafeMutableRawPointer) -> UnsafePointer<CChar>? {
	let model = Unmanaged<DepthModel>.fromOpaque(modelPtr).takeUnretainedValue().model
	let desc = model.modelDescription
	let info = "Model: \(desc.metadata[.description] ?? "Unknown")"
	return UnsafePointer(strdup(info))
}

final class DepthModel {
	let model: MLModel
	let stream: StreamWindow?

	init(model: MLModel, stream: StreamWindow?) {
		self.model = model
		self.stream = stream
	}
}

// Temporal cache for Video Depth Anything streaming models (inputs cache0..7, outputs state0..7).
// Mirrors VideoDepthAnything.infer_video_depth_one: frame 0 stays in the window for the whole video.
final class StreamWindow {
	static let windowLength = 31
	static let layerCount = 8
	static let dropAfterFrame = 10

	var frames: [[[Float16]]] = []
	var frameId = 0

	static func isStreaming(_ model: MLModel) -> Bool {
		model.modelDescription.inputDescriptionsByName["cache0"] != nil
	}

	func reset() {
		frames = []
		frameId = 0
	}

	func predict(_ model: MLModel, image: MLFeatureValue) throws -> MLFeatureProvider {
		let shapes = try Self.cacheShapes(model)
		if frames.isEmpty {
			let zeros = shapes.map { [Float16](repeating: 0, count: $0[1] * $0[2]) }
			let primed = try run(model, image: image, window: Array(repeating: zeros, count: Self.windowLength), shapes: shapes)
			let primedStates = try Self.states(primed, shapes: shapes)
			let result = try run(model, image: image, window: Array(repeating: primedStates, count: Self.windowLength), shapes: shapes)
			push(try Self.states(result, shapes: shapes))
			return result
		}
		let window = Array(frames[0..<2]) + Array(frames.suffix(Self.windowLength - 2))
		let result = try run(model, image: image, window: window, shapes: shapes)
		push(try Self.states(result, shapes: shapes))
		return result
	}

	private func push(_ states: [[Float16]]) {
		if frames.isEmpty {
			frames = Array(repeating: states, count: Self.windowLength + 1)
		} else {
			frames.append(states)
		}
		if frameId > Self.dropAfterFrame {
			frames.remove(at: 1)
		}
		frameId += 1
		assert(frames.count <= Self.windowLength + Self.dropAfterFrame + 2)
	}

	private func run(_ model: MLModel, image: MLFeatureValue, window: [[[Float16]]], shapes: [[Int]]) throws -> MLFeatureProvider {
		precondition(window.count == Self.windowLength)
		var inputs: [String: MLFeatureValue] = ["image": image]
		for layer in 0..<Self.layerCount {
			let shape = shapes[layer]
			let cache = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
			let frameSize = shape[1] * shape[2]
			let dst = cache.dataPointer.assumingMemoryBound(to: Float16.self)
			for (index, frame) in window.enumerated() {
				precondition(frame[layer].count == frameSize)
				frame[layer].withUnsafeBufferPointer { src in
					(dst + index * frameSize).update(from: src.baseAddress!, count: frameSize)
				}
			}
			inputs["cache\(layer)"] = MLFeatureValue(multiArray: cache)
		}
		return try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs))
	}

	private static func cacheShapes(_ model: MLModel) throws -> [[Int]] {
		try (0..<layerCount).map { layer in
			guard let shape = model.modelDescription.inputDescriptionsByName["cache\(layer)"]?
				.multiArrayConstraint?.shape.map({ $0.intValue }), shape.count == 3, shape[0] == windowLength
			else {
				throw NSError(domain: "CoreMLDepth", code: 1, userInfo: [NSLocalizedDescriptionKey: "bad cache\(layer) shape"])
			}
			return shape
		}
	}

	private static func states(_ result: MLFeatureProvider, shapes: [[Int]]) throws -> [[Float16]] {
		try (0..<layerCount).map { layer in
			guard let state = result.featureValue(for: "state\(layer)")?.multiArrayValue else {
				throw NSError(domain: "CoreMLDepth", code: 2, userInfo: [NSLocalizedDescriptionKey: "missing state\(layer)"])
			}
			let rows = shapes[layer][1]
			let cols = shapes[layer][2]
			let strides = state.strides.map { $0.intValue }
			let src = state.dataPointer.assumingMemoryBound(to: Float16.self)
			var out = [Float16](repeating: 0, count: rows * cols)
			for row in 0..<rows {
				for col in 0..<cols {
					out[row * cols + col] = src[row * strides[1] + col * strides[2]]
				}
			}
			return out
		}
	}
}
