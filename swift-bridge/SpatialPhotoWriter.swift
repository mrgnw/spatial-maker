import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

enum PhotoWriterError: Error {
	case image(String)
	case destination(String)
}

extension SpatialMetadata {
	func photoProperties(isLeft: Bool, width: Int, height: Int, quality: Double) -> [CFString: Any] {
		let baselineMeters = Double(baselineMicrometers) / 1_000_000.0
		let fovRadians = Double(horizontalFOVMillidegrees) / 1000.0 / 180.0 * .pi
		let focalLength = (Double(width) * 0.5) / tan(fovRadians * 0.5)
		let intrinsics: [Double] = [
			focalLength, 0, Double(width) * 0.5,
			0, focalLength, Double(height) * 0.5,
			0, 0, 1,
		]
		let identityRotation: [Double] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
		let position: [Double] = isLeft ? [0, 0, 0] : [baselineMeters, 0, 0]

		return [
			kCGImagePropertyGroups: [
				kCGImagePropertyGroupIndex: 0,
				kCGImagePropertyGroupType: kCGImagePropertyGroupTypeStereoPair,
				(isLeft ? kCGImagePropertyGroupImageIsLeftImage : kCGImagePropertyGroupImageIsRightImage): true,
				kCGImagePropertyGroupImageDisparityAdjustment: Int(disparityAdjustment),
			],
			kCGImagePropertyHEIFDictionary: [
				kIIOMetadata_CameraExtrinsicsKey: [
					kIIOCameraExtrinsics_Position: position,
					kIIOCameraExtrinsics_Rotation: identityRotation,
				],
				kIIOMetadata_CameraModelKey: [
					kIIOCameraModel_Intrinsics: intrinsics,
					kIIOCameraModel_ModelType: kIIOCameraModelType_SimplifiedPinhole,
				],
			],
			kCGImagePropertyHasAlpha: false,
			kCGImageDestinationLossyCompressionQuality: quality,
		]
	}
}

func makeRGBImage(_ pixels: UnsafePointer<UInt8>, width: Int, height: Int) throws -> CGImage {
	let bytesPerRow = width * 3
	let data = Data(bytes: pixels, count: bytesPerRow * height)
	guard let provider = CGDataProvider(data: data as CFData),
		let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
		let image = CGImage(
			width: width,
			height: height,
			bitsPerComponent: 8,
			bitsPerPixel: 24,
			bytesPerRow: bytesPerRow,
			space: colorSpace,
			bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
			provider: provider,
			decode: nil,
			shouldInterpolate: false,
			intent: .defaultIntent
		)
	else {
		throw PhotoWriterError.image("cannot wrap \(width)x\(height) RGB pixels as CGImage")
	}
	return image
}

func writeSpatialPhoto(
	outputURL: URL,
	left: UnsafePointer<UInt8>,
	right: UnsafePointer<UInt8>,
	width: Int,
	height: Int,
	quality: Double,
	spatial: SpatialMetadata
) throws {
	precondition(width > 0 && height > 0)
	precondition(quality >= 0 && quality <= 1)
	let leftImage = try makeRGBImage(left, width: width, height: height)
	let rightImage = try makeRGBImage(right, width: width, height: height)

	guard let destination = CGImageDestinationCreateWithURL(
		outputURL as CFURL,
		UTType.heic.identifier as CFString,
		2,
		[kCGImagePropertyPrimaryImage: 0] as CFDictionary
	) else {
		throw PhotoWriterError.destination("cannot create HEIC destination at \(outputURL.path)")
	}
	CGImageDestinationAddImage(
		destination,
		leftImage,
		spatial.photoProperties(isLeft: true, width: width, height: height, quality: quality) as CFDictionary
	)
	CGImageDestinationAddImage(
		destination,
		rightImage,
		spatial.photoProperties(isLeft: false, width: width, height: height, quality: quality) as CFDictionary
	)
	guard CGImageDestinationFinalize(destination) else {
		throw PhotoWriterError.destination("cannot finalize HEIC at \(outputURL.path)")
	}
}

@_cdecl("spatial_photo_write")
public func spatialPhotoWrite(
	_ outputPathPtr: UnsafePointer<CChar>,
	_ left: UnsafePointer<UInt8>,
	_ right: UnsafePointer<UInt8>,
	_ width: Int32,
	_ height: Int32,
	_ quality: Double,
	_ baselineMicrometers: UInt32,
	_ horizontalFOVMillidegrees: UInt32,
	_ disparityAdjustment: Int32
) -> Int32 {
	let spatial = SpatialMetadata(
		baselineMicrometers: baselineMicrometers,
		horizontalFOVMillidegrees: horizontalFOVMillidegrees,
		disparityAdjustment: disparityAdjustment
	)
	do {
		try writeSpatialPhoto(
			outputURL: URL(fileURLWithPath: String(cString: outputPathPtr)),
			left: left,
			right: right,
			width: Int(width),
			height: Int(height),
			quality: quality,
			spatial: spatial
		)
		return 0
	} catch PhotoWriterError.image {
		return -1
	} catch {
		return -2
	}
}
