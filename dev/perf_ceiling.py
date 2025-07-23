

# Generally assumed padding="same"
def mem_conv(S,R,H,W,C,K, typeSize):
    mem_in = H * W * C * typeSize
    mem_out = H * W * K * typeSize
    mem_filter = S * R * C * K * typeSize
    mem_bias = K * typeSize
    return mem_in + mem_out + mem_filter + mem_bias

def mem_pool(H,W,C, typeSize):
    mem_in = H * W * C * typeSize
    mem_out = (H // 2) * (W // 2) * C * typeSize
    mem = mem_in + mem_out
    return mem

def mem_upsample(H,W,C,typeSize):
    mem_in = H * W * C * typeSize
    mem_out = H * 2 * W * 2 * C * typeSize
    mem = mem_in + mem_out
    return mem


W = 1920
H = 1080
CONV_MEMORY_BANDWIDTH = 300e9
POOL_MEMORY_BANDWIDTH = 400e9
UPSAMPLE_MEMORY_BANDWIDTH = 400e9

FUSE_POOL_MEM_FACTOR = 1
FUSE_UPSAMPLE_MEM_FACTOR = 1
mem = 0;
f16 = 2


mem_enc0 = mem_conv(3,3,W,H, 3, 32, f16)
mem_pool0 = mem_pool(W,H,32,f16) / FUSE_POOL_MEM_FACTOR

mem_enc1 = mem_conv(3,3,W // 2, H // 2, 32, 48, f16)
mem_pool1 = mem_pool(W // 2,H // 2, 48, f16) / FUSE_POOL_MEM_FACTOR

mem_enc2 = mem_conv(3,3,W // 4, H // 4, 48, 64, f16)
mem_pool2 = mem_pool(W // 4, H // 4, 64, f16) / FUSE_POOL_MEM_FACTOR

mem_enc3 = mem_conv(3,3,W // 8, H // 8, 64, 80, f16)
mem_pool3 = mem_pool(W // 8, H // 8, 80, f16) / FUSE_POOL_MEM_FACTOR

mem_enc4 = mem_conv(3,3, W // 16, H // 16, 80, 112, f16)
mem_pool4 = mem_pool(W // 16, H // 16, 112, f16) / FUSE_POOL_MEM_FACTOR

mem_enc5 = mem_conv(3,3, W // 16, H // 16, 112, 112, f16)
mem_pool5 = mem_pool(W // 16, H // 16, 112, f16) / FUSE_POOL_MEM_FACTOR
# pool
# upsample + concat
mem_up0 = mem_upsample(W // 64, W // 64, 112, f16) / FUSE_UPSAMPLE_MEM_FACTOR;
mem_dec0 = mem_conv(3,3, W // 32, H // 32, 112 + 112, 112, f16)
mem_con0 = mem_conv(3,3, W // 32, H // 32, 112, 112, f16)
# upsample + concat
mem_up1 = mem_upsample(W // 32, W // 32, 112, f16) / FUSE_UPSAMPLE_MEM_FACTOR;
mem_dec1 = mem_conv(3,3, W // 16, H // 16, 112 + 80, 80, f16)
mem_con1 = mem_conv(3,3, W // 16, H // 16, 80, 80, f16)
# upsample + concat
mem_up2 = mem_upsample(W // 16, W // 16, 80, f16) / FUSE_UPSAMPLE_MEM_FACTOR;
mem_dec2 = mem_conv(3,3, W // 8, H // 8, 80 + 64, 64, f16)
mem_con2 = mem_conv(3,3, W // 8, H // 8, 64, 64, f16)
# upsample + concat
mem_up3 = mem_upsample(W // 8, W // 8, 64, f16) / FUSE_UPSAMPLE_MEM_FACTOR;
mem_dec3 = mem_conv(3,3, W // 4, H // 4, 64 + 48, 48, f16)
mem_con3 = mem_conv(3,3, W // 4, H // 4, 48, 48, f16)
# upsample + concat
mem_up4 = mem_upsample(W // 4, W // 4, 48, f16) / FUSE_UPSAMPLE_MEM_FACTOR;
mem_dec4 = mem_conv(3,3, W // 2, H // 2, 48 + 32, 16, f16)
mem_con4 = mem_conv(3,3, W // 2, H // 2, 16, 16, f16)
# upsample + concat
mem_up5 = mem_upsample(W // 2, W // 2, 16, f16) / FUSE_UPSAMPLE_MEM_FACTOR;
mem_dec5 = mem_conv(3,3, W // 1, H // 1, 16 + 3, 12, f16)
mem_con5 = mem_conv(3,3, W // 1, H // 1, 12, 12, f16)

lat_enc0 = mem_enc0 / CONV_MEMORY_BANDWIDTH
lat_enc1 = mem_enc1 / CONV_MEMORY_BANDWIDTH
lat_enc2 = mem_enc2 / CONV_MEMORY_BANDWIDTH
lat_enc3 = mem_enc3 / CONV_MEMORY_BANDWIDTH
lat_enc4 = mem_enc4 / CONV_MEMORY_BANDWIDTH
lat_enc5 = mem_enc5 / CONV_MEMORY_BANDWIDTH

lat_dec0 = mem_dec0 / CONV_MEMORY_BANDWIDTH
lat_dec1 = mem_dec1 / CONV_MEMORY_BANDWIDTH
lat_dec2 = mem_dec2 / CONV_MEMORY_BANDWIDTH
lat_dec3 = mem_dec3 / CONV_MEMORY_BANDWIDTH
lat_dec4 = mem_dec4 / CONV_MEMORY_BANDWIDTH
lat_dec5 = mem_dec5 / CONV_MEMORY_BANDWIDTH

lat_con0 = mem_con0 / CONV_MEMORY_BANDWIDTH
lat_con1 = mem_con1 / CONV_MEMORY_BANDWIDTH
lat_con2 = mem_con2 / CONV_MEMORY_BANDWIDTH
lat_con3 = mem_con3 / CONV_MEMORY_BANDWIDTH
lat_con4 = mem_con4 / CONV_MEMORY_BANDWIDTH
lat_con5 = mem_con5 / CONV_MEMORY_BANDWIDTH

lat_pool0 = mem_pool0 / POOL_MEMORY_BANDWIDTH
lat_pool1 = mem_pool1 / POOL_MEMORY_BANDWIDTH
lat_pool2 = mem_pool2 / POOL_MEMORY_BANDWIDTH
lat_pool3 = mem_pool3 / POOL_MEMORY_BANDWIDTH
lat_pool4 = mem_pool4 / POOL_MEMORY_BANDWIDTH
lat_pool5 = mem_pool5 / POOL_MEMORY_BANDWIDTH

lat_up0 = mem_up0 / UPSAMPLE_MEMORY_BANDWIDTH
lat_up1 = mem_up1 / UPSAMPLE_MEMORY_BANDWIDTH
lat_up2 = mem_up2 / UPSAMPLE_MEMORY_BANDWIDTH
lat_up3 = mem_up3 / UPSAMPLE_MEMORY_BANDWIDTH
lat_up4 = mem_up4 / UPSAMPLE_MEMORY_BANDWIDTH
lat_up5 = mem_up5 / UPSAMPLE_MEMORY_BANDWIDTH


print("mem_enc0:  {:5.5}MB   latency: {:3.3}ms".format(mem_enc0 * 1e-6, lat_enc0 * 1e3))
print("mem_pool0: {:5.5}MB   latency: {:3.3}ms".format(mem_pool0 * 1e-6, lat_pool0 * 1e3))

print("mem_enc1:  {:5.5}MB   latency: {:3.3}ms".format(mem_enc1 * 1e-6, lat_enc1 * 1e3))
print("mem_pool1: {:5.5}MB   latency: {:3.3}ms".format(mem_pool1 * 1e-6, lat_pool1 * 1e3))

print("mem_enc2:  {:5.5}MB   latency: {:3.3}ms".format(mem_enc2 * 1e-6, lat_enc2 * 1e3))
print("mem_pool2: {:5.5}MB   latency: {:3.3}ms".format(mem_pool2 * 1e-6, lat_pool2 * 1e3))

print("mem_enc3:  {:5.5}MB   latency: {:3.3}ms".format(mem_enc3 * 1e-6, lat_enc3 * 1e3))
print("mem_pool3: {:5.5}MB   latency: {:3.3}ms".format(mem_pool3 * 1e-6, lat_pool3 * 1e3))

print("mem_enc4:  {:5.5}MB   latency: {:3.3}ms".format(mem_enc4 * 1e-6, lat_enc4 * 1e3))
print("mem_pool4: {:5.5}MB   latency: {:3.3}ms".format(mem_pool4 * 1e-6, lat_pool4 * 1e3))

print("mem_enc5:  {:5.5}MB   latency: {:3.3}ms".format(mem_enc5 * 1e-6, lat_enc5 * 1e3))
print("mem_pool5: {:5.5}MB   latency: {:3.3}ms".format(mem_pool5 * 1e-6, lat_pool5 * 1e3))





print("mem_up0:   {:5.5}MB    latency: {:3.3}ms".format(mem_up0 * 1e-6, lat_up0 * 1e3))
print("mem_dec0:  {:5.5}MB   latency: {:3.3}ms".format(mem_dec0 * 1e-6, lat_dec0 * 1e3))
print("mem_con0:  {:5.5}MB   latency: {:3.3}ms".format(mem_con0 * 1e-6, lat_con0 * 1e3))

print("mem_up1:   {:5.5}MB    latency: {:3.3}ms".format(mem_up1 * 1e-6, lat_up1 * 1e3))
print("mem_dec1:  {:5.5}MB   latency: {:3.3}ms".format(mem_dec1 * 1e-6, lat_dec1 * 1e3))
print("mem_con1:  {:5.5}MB   latency: {:3.3}ms".format(mem_con1 * 1e-6, lat_con1 * 1e3))

print("mem_up2:   {:5.5}MB    latency: {:3.3}ms".format(mem_up2 * 1e-6, lat_up2 * 1e3))
print("mem_dec2:  {:5.5}MB   latency: {:3.3}ms".format(mem_dec2 * 1e-6, lat_dec2 * 1e3))
print("mem_con2:  {:5.5}MB   latency: {:3.3}ms".format(mem_con2 * 1e-6, lat_con2 * 1e3))

print("mem_up3:   {:5.5}MB   latency: {:3.3}ms".format(mem_up3 * 1e-6, lat_up3 * 1e3))
print("mem_dec3:  {:5.5}MB   latency: {:3.3}ms".format(mem_dec3 * 1e-6, lat_dec3 * 1e3))
print("mem_con3:  {:5.5}MB   latency: {:3.3}ms".format(mem_con3 * 1e-6, lat_con3 * 1e3))

print("mem_up4:   {:5.5}MB   latency: {:3.3}ms".format(mem_up4 * 1e-6, lat_up4 * 1e3))
print("mem_dec4:  {:5.5}MB   latency: {:3.3}ms".format(mem_dec4 * 1e-6, lat_dec4 * 1e3))
print("mem_con4:  {:5.5}MB   latency: {:3.3}ms".format(mem_con4 * 1e-6, lat_con4 * 1e3))

print("mem_up5:   {:5.5}MB   latency: {:3.3}ms".format(mem_up5 * 1e-6, lat_up5 * 1e3))
print("mem_dec5:  {:5.5}MB   latency: {:3.3}ms".format(mem_dec5 * 1e-6, lat_dec5 * 1e3))
print("mem_con5:  {:5.5}MB   latency: {:3.3}ms".format(mem_con5 * 1e-6, lat_con5 * 1e3))

lat = lat_enc0 + lat_enc1 + lat_enc2 + lat_enc3 + lat_enc4 + lat_enc5 + lat_dec0 + lat_dec1 + lat_dec2 + lat_dec3 + lat_dec4 + lat_dec5 + lat_con0 + lat_con1 + lat_con2 + lat_con3 + lat_con4 + lat_con5 + lat_pool0 + lat_pool1 + lat_pool2 + lat_pool3 + lat_pool4 + lat_pool5 + lat_up0 + lat_up1 + lat_up2 + lat_up3 + lat_up4 + lat_up5;

print("Total-Latency {:3.3}ms".format(lat * 1e3));

