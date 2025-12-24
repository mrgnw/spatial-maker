import os
from depth_estimators import AppleDepthPro, DepthAnythingV2Estimator

num_threads = max(1, os.cpu_count() - 4)

print('\n' + '='*60)
print('Benchmarking Apple Depth Pro')
print('='*60)
estimator = AppleDepthPro()
result = estimator.process_video('samples/1080p24fps/barcelona-lights_1080p.MOV', 'output/apple_depth_pro')
print('RESULT:', result)

# print('\n' + '='*60)
# print('Benchmarking Depth Anything V2 Small')
# print('='*60)
# estimator = DepthAnythingV2Estimator(encoder='vits', num_threads=num_threads)
# result = estimator.process_video('samples/1080p24fps/barcelona-lights_1080p.MOV', 'output/dav2_small')
# print('RESULT:', result)

# print('\n' + '='*60)
# print('Benchmarking Depth Anything V2 Base')
# print('='*60)
# estimator = DepthAnythingV2Estimator(encoder='vitb', num_threads=num_threads)
# result = estimator.process_video('samples/1080p24fps/barcelona-lights_1080p.MOV', 'output/dav2_base')
# print('RESULT:', result)

# print('\n' + '='*60)
# print('Benchmarking Depth Anything V2 Large')
# print('='*60)
# estimator = DepthAnythingV2Estimator(encoder='vitl', num_threads=num_threads)
# result = estimator.process_video('samples/1080p24fps/barcelona-lights_1080p.MOV', 'output/dav2_large')
# print('RESULT:', result)
