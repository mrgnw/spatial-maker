import os
from depth_estimators import AppleDepthPro, DepthAnythingV2Estimator

num_threads = max(1, os.cpu_count() - 4)

print('\n' + '='*60)
print('Benchmarking Depth Anything V2 Small')
print('='*60)
estimator = DepthAnythingV2Estimator(encoder='vits', num_threads=num_threads)
result = estimator.process_video('samples/1080p24fps/barcelona-lights_1080p.MOV', 'output/depth_anything_v2_small_bench')
print('RESULT:', result)

print('\n' + '='*60)
print('Benchmarking Depth Anything V2 Base')
print('='*60)
estimator = DepthAnythingV2Estimator(encoder='vitb', num_threads=num_threads)
result = estimator.process_video('samples/1080p24fps/barcelona-lights_1080p.MOV', 'output/depth_anything_v2_base_bench')
print('RESULT:', result)

print('\n' + '='*60)
print('Benchmarking Depth Anything V2 Large')
print('='*60)
estimator = DepthAnythingV2Estimator(encoder='vitl', num_threads=num_threads)
result = estimator.process_video('samples/1080p24fps/barcelona-lights_1080p.MOV', 'output/depth_anything_v2_large_bench')
print('RESULT:', result)
