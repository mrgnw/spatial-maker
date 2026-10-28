import CoreVideo
import Foundation
import Vision

// Soft matte (0...1) of every foreground instance Vision finds, at the image's own resolution.
// Returns the instance count: 0 means no subject, negative means Vision is unavailable or failed.
@_cdecl("vision_foreground_matte")
public func foregroundMatte(
	_ rgbData: UnsafePointer<UInt8>,
	_ width: Int32,
	_ height: Int32,
	_ outputPtr: UnsafeMutablePointer<Float>
) -> Int32 {
	guard #available(macOS 14.0, *) else { return -1 }
	let w = Int(width)
	let h = Int(height)
	precondition(w > 0 && h > 0)

	do {
		let image = try makeRGBImage(rgbData, width: w, height: h)
		let handler = VNImageRequestHandler(cgImage: image)
		let request = VNGenerateForegroundInstanceMaskRequest()
		try handler.perform([request])
		guard let observation = request.results?.first, !observation.allInstances.isEmpty else {
			return 0
		}
		let mask = try observation.generateScaledMaskForImage(
			forInstances: observation.allInstances,
			from: handler
		)
		guard CVPixelBufferGetWidth(mask) == w, CVPixelBufferGetHeight(mask) == h,
			CVPixelBufferGetPixelFormatType(mask) == kCVPixelFormatType_OneComponent32Float
		else {
			print("Vision matte has unexpected shape or format")
			return -3
		}

		CVPixelBufferLockBaseAddress(mask, .readOnly)
		defer { CVPixelBufferUnlockBaseAddress(mask, .readOnly) }
		let base = CVPixelBufferGetBaseAddress(mask)!
		let bytesPerRow = CVPixelBufferGetBytesPerRow(mask)
		for y in 0..<h {
			let row = (base + y * bytesPerRow).assumingMemoryBound(to: Float.self)
			(outputPtr + y * w).update(from: row, count: w)
		}
		return Int32(observation.allInstances.count)
	} catch {
		print("Vision foreground matte failed: \(error)")
		return -2
	}
}
