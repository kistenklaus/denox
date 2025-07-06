


## Begin Input

C = 128
K = 128

S = 3
R = 3

mmaShapeN = 16
mmaShapeK = 16
mmaShapeM = 16

workgroupSize = 256
subgroupSize = 32

pipelineDepth = 1
floatByteSize = 2


## End Input



tile_h = workgroupSize // subgroupSize
lowered_input_rows = tile_h * mmaShapeN
lowered_input_cols = R * S * C
tile_w = lowered_input_rows // tile_h

lowered_input_rows_with_h_reuse = tile_h * (tile_w + (S-1))

print()
print("Input tile size:", (tile_w, tile_h))

print("Lowered input feature map size:", (lowered_input_rows, lowered_input_cols))

print("Lowered input feature map tile size (without reuse):", (pipelineDepth, lowered_input_rows, mmaShapeK))
lowered_tile_footprint_without_reuse = (lowered_input_rows * mmaShapeK * pipelineDepth * floatByteSize);
print(f'Lowered input feature map tile shared footprint (without reuse): {(lowered_tile_footprint_without_reuse) / 1024}KB')


lowered_tile_footprint_with_h_reuse = (lowered_input_rows_with_h_reuse * mmaShapeK * pipelineDepth * floatByteSize)
print("Lowered input feature map tile size (with h-reuse):", (pipelineDepth, lowered_input_rows_with_h_reuse, mmaShapeK))
print(f'Lowered input feature map tile shared footprint (with h-reuse): {lowered_tile_footprint_with_h_reuse / 1024}KB')


print()
print("Weight matrix size:", (lowered_input_cols, K))

print("Weight matrix tile size (without reuse):", (pipelineDepth, mmaShapeK, K));
weight_tile_footprint_without_reuse = pipelineDepth * mmaShapeK * K * floatByteSize
print(f'Weight matrix tile shared footprint (without reuse): {weight_tile_footprint_without_reuse / 1024}KB')

print("Weight matrix tile size (with h-reuse):", (pipelineDepth, S, mmaShapeK, K));
weight_tile_footprint_with_h_reuse = (S * pipelineDepth * mmaShapeK * K * floatByteSize)
print(f'Weight matrix tile shared footprint (wit h-reuse): { weight_tile_footprint_with_h_reuse/ 1024}KB')


print()
shared_footprint_without_reuse = (lowered_tile_footprint_without_reuse + weight_tile_footprint_without_reuse)
print(f'Total shared memory footprint (without reuse): {shared_footprint_without_reuse / 1024}KB')
print(f'Max workgroups limited by shared (without reuse) : {100e3 // shared_footprint_without_reuse}')

shared_footprint_with_h_reuse = (lowered_tile_footprint_with_h_reuse + weight_tile_footprint_with_h_reuse)
print(f'Total shared memory footprint (with h-reuse): {shared_footprint_with_h_reuse / 1024}KB')
print(f'Max workgroups limited by shared (with h-reuse) : {100e3 // shared_footprint_with_h_reuse}')



regwithKC32 = 37
