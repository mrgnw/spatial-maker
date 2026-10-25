import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import VideoToolbox

let appendTimeoutSeconds = 30.0

@available(macOS 14.0, *)
final class SpatialVideoWriter {
	let writer: AVAssetWriter
	let videoInput: AVAssetWriterInput
	let adaptor: AVAssetWriterInputTaggedPixelBufferGroupAdaptor
	let width: Int
	let height: Int
	let fps: Double
	var frameIndex: Int64 = 0
	var audio: AudioPassthrough?

	init(outputURL: URL, audioSourceURL: URL?, width: Int, height: Int, fps: Double, spatial: SpatialMetadata) throws {
		precondition(width > 0 && height > 0 && fps > 0)
		try? FileManager.default.removeItem(at: outputURL)
		writer = try AVAssetWriter(outputURL: outputURL, fileType: .mov)
		self.width = width
		self.height = height
		self.fps = fps

		videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: [
			AVVideoCodecKey: AVVideoCodecType.hevc,
			AVVideoWidthKey: width,
			AVVideoHeightKey: height,
			AVVideoColorPropertiesKey: [
				AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
				AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
				AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
			],
			AVVideoCompressionPropertiesKey: spatial.compressionProperties(),
		])
		videoInput.expectsMediaDataInRealTime = false
		adaptor = AVAssetWriterInputTaggedPixelBufferGroupAdaptor(
			assetWriterInput: videoInput,
			sourcePixelBufferAttributes: [
				kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
				kCVPixelBufferWidthKey as String: width,
				kCVPixelBufferHeightKey as String: height,
			]
		)
		guard writer.canAdd(videoInput) else { throw WriterError.setup("cannot add MV-HEVC video input") }
		writer.add(videoInput)

		if let audioSourceURL {
			audio = try AudioPassthrough(sourceURL: audioSourceURL, writer: writer)
		}

		guard writer.startWriting() else { throw WriterError.setup("startWriting: \(String(describing: writer.error))") }
		writer.startSession(atSourceTime: .zero)
		audio?.start()
	}

	func append(left: UnsafePointer<UInt8>, right: UnsafePointer<UInt8>) throws {
		let leftBuffer = try pixelBuffer(rgb: left)
		let rightBuffer = try pixelBuffer(rgb: right)
		let tagged = [
			CMTaggedBuffer(tags: [.stereoView(.leftEye), .videoLayerID(0)], pixelBuffer: leftBuffer),
			CMTaggedBuffer(tags: [.stereoView(.rightEye), .videoLayerID(1)], pixelBuffer: rightBuffer),
		]
		try waitUntilReady()
		let time = CMTime(seconds: Double(frameIndex) / fps, preferredTimescale: 90000)
		guard adaptor.appendTaggedBuffers(tagged, withPresentationTime: time) else {
			throw WriterError.append("frame \(frameIndex): \(String(describing: writer.error))")
		}
		frameIndex += 1
	}

	func finish() throws {
		videoInput.markAsFinished()
		audio?.waitUntilDone()
		let done = DispatchSemaphore(value: 0)
		writer.finishWriting { done.signal() }
		done.wait()
		guard writer.status == .completed else {
			throw WriterError.finish("\(writer.status.rawValue): \(String(describing: writer.error))")
		}
	}

	private func waitUntilReady() throws {
		let deadline = Date().addingTimeInterval(appendTimeoutSeconds)
		while !videoInput.isReadyForMoreMediaData {
			guard writer.status == .writing else { throw WriterError.append("writer stopped: \(String(describing: writer.error))") }
			guard Date() < deadline else { throw WriterError.append("video input not ready after \(appendTimeoutSeconds)s") }
			usleep(1000)
		}
	}

	private func pixelBuffer(rgb: UnsafePointer<UInt8>) throws -> CVPixelBuffer {
		guard let pool = adaptor.pixelBufferPool else { throw WriterError.append("no pixel buffer pool") }
		var buffer: CVPixelBuffer?
		guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer) == kCVReturnSuccess, let buffer else {
			throw WriterError.append("pixel buffer allocation failed")
		}
		CVPixelBufferLockBaseAddress(buffer, [])
		defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
		let bgra = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
		let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
		for y in 0..<height {
			let src = rgb + y * width * 3
			let dst = bgra + y * bytesPerRow
			for x in 0..<width {
				dst[x * 4 + 0] = src[x * 3 + 2]
				dst[x * 4 + 1] = src[x * 3 + 1]
				dst[x * 4 + 2] = src[x * 3 + 0]
				dst[x * 4 + 3] = 255
			}
		}
		return buffer
	}
}

struct SpatialMetadata {
	let baselineMicrometers: UInt32
	let horizontalFOVMillidegrees: UInt32
	let disparityAdjustment: Int32

	@available(macOS 14.0, *)
	func compressionProperties() -> [CFString: Any] {
		var properties: [CFString: Any] = [
			kVTCompressionPropertyKey_MVHEVCVideoLayerIDs: [0, 1],
			kVTCompressionPropertyKey_MVHEVCViewIDs: [0, 1],
			kVTCompressionPropertyKey_MVHEVCLeftAndRightViewIDs: [0, 1],
			kVTCompressionPropertyKey_HasLeftStereoEyeView: true,
			kVTCompressionPropertyKey_HasRightStereoEyeView: true,
			kVTCompressionPropertyKey_HeroEye: kCMFormatDescriptionHeroEye_Left,
			kVTCompressionPropertyKey_StereoCameraBaseline: baselineMicrometers,
			kVTCompressionPropertyKey_HorizontalDisparityAdjustment: disparityAdjustment,
			kVTCompressionPropertyKey_MaxKeyFrameIntervalDuration: 2.0,
		]
		if #available(macOS 14.4, *) {
			properties[kVTCompressionPropertyKey_HorizontalFieldOfView] = horizontalFOVMillidegrees
		}
		if #available(macOS 15.0, *) {
			properties[kVTCompressionPropertyKey_ProjectionKind] = kCMFormatDescriptionProjectionKind_Rectilinear
		}
		return properties
	}
}

@available(macOS 14.0, *)
final class AudioPassthrough {
	let reader: AVAssetReader
	let output: AVAssetReaderTrackOutput
	let input: AVAssetWriterInput
	let queue = DispatchQueue(label: "spatial-maker.audio")
	let done = DispatchGroup()

	init(sourceURL: URL, writer: AVAssetWriter) throws {
		let asset = AVURLAsset(url: sourceURL)
		let tracks = try loadAudioTracks(asset)
		guard let track = tracks.first else { throw WriterError.setup("no audio track in \(sourceURL.path)") }
		reader = try AVAssetReader(asset: asset)
		output = AVAssetReaderTrackOutput(track: track, outputSettings: [
			AVFormatIDKey: kAudioFormatLinearPCM,
			AVLinearPCMBitDepthKey: 16,
			AVLinearPCMIsFloatKey: false,
			AVLinearPCMIsBigEndianKey: false,
			AVLinearPCMIsNonInterleaved: false,
			AVSampleRateKey: 48000,
			AVNumberOfChannelsKey: 2,
		])
		guard reader.canAdd(output) else { throw WriterError.setup("cannot read audio track") }
		reader.add(output)
		input = AVAssetWriterInput(mediaType: .audio, outputSettings: [
			AVFormatIDKey: kAudioFormatMPEG4AAC,
			AVSampleRateKey: 48000,
			AVNumberOfChannelsKey: 2,
			AVEncoderBitRateKey: 192_000,
		])
		input.expectsMediaDataInRealTime = false
		guard writer.canAdd(input) else { throw WriterError.setup("cannot add audio input") }
		writer.add(input)
	}

	func start() {
		guard reader.startReading() else {
			input.markAsFinished()
			return
		}
		done.enter()
		input.requestMediaDataWhenReady(on: queue) { [self] in
			while input.isReadyForMoreMediaData {
				guard let sample = output.copyNextSampleBuffer(), input.append(sample) else {
					input.markAsFinished()
					reader.cancelReading()
					done.leave()
					return
				}
			}
		}
	}

	func waitUntilDone() {
		done.wait()
	}
}

@available(macOS 14.0, *)
func loadAudioTracks(_ asset: AVURLAsset) throws -> [AVAssetTrack] {
	let loaded = DispatchSemaphore(value: 0)
	var tracks: [AVAssetTrack]?
	var loadError: Error?
	asset.loadTracks(withMediaType: .audio) { loadedTracks, error in
		tracks = loadedTracks
		loadError = error
		loaded.signal()
	}
	loaded.wait()
	if let loadError { throw loadError }
	return tracks ?? []
}

enum WriterError: Error {
	case setup(String)
	case append(String)
	case finish(String)
}

// MARK: - C API for Rust FFI

@_cdecl("spatial_writer_create")
public func spatialWriterCreate(
	_ outputPathPtr: UnsafePointer<CChar>,
	_ audioSourcePathPtr: UnsafePointer<CChar>?,
	_ width: Int32,
	_ height: Int32,
	_ fps: Double,
	_ baselineMicrometers: UInt32,
	_ horizontalFOVMillidegrees: UInt32,
	_ disparityAdjustment: Int32
) -> UnsafeMutableRawPointer? {
	guard #available(macOS 14.0, *) else {
		print("MV-HEVC encoding needs macOS 14 or later")
		return nil
	}
	let outputURL = URL(fileURLWithPath: String(cString: outputPathPtr))
	let audioURL = audioSourcePathPtr.map { URL(fileURLWithPath: String(cString: $0)) }
	let spatial = SpatialMetadata(
		baselineMicrometers: baselineMicrometers,
		horizontalFOVMillidegrees: horizontalFOVMillidegrees,
		disparityAdjustment: disparityAdjustment
	)
	do {
		let writer = try SpatialVideoWriter(
			outputURL: outputURL, audioSourceURL: audioURL,
			width: Int(width), height: Int(height), fps: fps, spatial: spatial
		)
		return Unmanaged.passRetained(writer).toOpaque()
	} catch {
		print("spatial writer: \(error)")
		return nil
	}
}

@_cdecl("spatial_writer_append")
public func spatialWriterAppend(
	_ writerPtr: UnsafeMutableRawPointer,
	_ left: UnsafePointer<UInt8>,
	_ right: UnsafePointer<UInt8>
) -> Int32 {
	guard #available(macOS 14.0, *) else { return -1 }
	let writer = Unmanaged<SpatialVideoWriter>.fromOpaque(writerPtr).takeUnretainedValue()
	do {
		try writer.append(left: left, right: right)
		return 0
	} catch {
		print("spatial writer: \(error)")
		return -2
	}
}

@_cdecl("spatial_writer_finish")
public func spatialWriterFinish(_ writerPtr: UnsafeMutableRawPointer) -> Int32 {
	guard #available(macOS 14.0, *) else { return -1 }
	let writer = Unmanaged<SpatialVideoWriter>.fromOpaque(writerPtr).takeRetainedValue()
	do {
		try writer.finish()
		return 0
	} catch {
		print("spatial writer: \(error)")
		return -2
	}
}
