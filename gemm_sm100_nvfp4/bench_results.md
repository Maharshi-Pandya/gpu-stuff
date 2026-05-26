# NVFP4 GEMM Benchmark: CuTeDSL vs cuBLAS (B200)

| (MxNxK) | Match | CuTeDSL (ms) | CuT TFLOPS | CuB (ms) | CuB TFLOPS | CuB / CuT |
| --- | --- | --- | --- | --- | --- | --- |
| 128x128x256 | True | 0.0082 | 1.0227 | 0.0298 | 0.2818 | 3.629x |
| 256x256x512 | True | 0.0082 | 8.1407 | 0.0292 | 2.2948 | 3.547x |
| 128x1024x1024 | True | 0.0084 | 31.9185 | 0.0309 | 8.6886 | 3.674x |
| 128x6144x12288 | True | 0.0225 | 858.1543 | 0.0476 | 405.6296 | 2.116x |
| 512x6144x6144 | True | 0.0225 | 1716.2395 | 0.0407 | 949.8336 | 1.807x |
| 1024x1024x1024 | True | 0.0083 | 260.2532 | 0.0305 | 70.3233 | 3.701x |
| 2048x2048x2048 | True | 0.0143 | 1197.3861 | 0.0335 | 513.5981 | 2.331x |
| 4096x4096x4096 | True | 0.0472 | 2913.9038 | 0.0543 | 2531.6418 | 1.151x |
| 8192x8192x8192 | True | 0.2868 | 3833.5003 | 0.2194 | 5010.5859 | 0.765x |


# NVFP4 MLP Benchmark: CuTeDSL vs scaled_mm (B200)

`fn(x) = act(lin3(act(lin2(act(lin1(x))))))`. Per-linear weight quant, weight scales, weight global scale, activation global scale, and output global scale are all fixed (calibrated once). Only activation block scales are recomputed per call. Both paths compiled with `dynamic=True, fullgraph=True, mode='max-autotune-no-cudagraphs'`.

| (MxNxK) | Match | CuTeDSL (ms) | CuT TFLOPS | ScaledMM (ms) | SMM TFLOPS | SMM / CuT |
| --- | --- | --- | --- | --- | --- | --- |
| 128x128x256 | True | 0.1450 | 0.1736 | 0.2035 | 0.1237 | 1.404x |
| 256x256x512 | True | 0.1417 | 1.4210 | 0.1991 | 1.0111 | 1.405x |
| 128x1024x1024 | True | 0.1469 | 5.4818 | 0.2119 | 3.8001 | 1.443x |
| 128x6144x12288 | True | 0.1494 | 388.1777 | 0.2292 | 252.9297 | 1.535x |
| 512x6144x6144 | True | 0.1491 | 777.8149 | 0.2078 | 557.9331 | 1.394x |
| 1024x1024x1024 | True | 0.1451 | 44.4081 | 0.2070 | 31.1184 | 1.427x |
| 2048x2048x2048 | True | 0.1515 | 340.1687 | 0.2077 | 248.1952 | 1.371x |
| 4096x4096x4096 | True | 0.2411 | 1710.3274 | 0.2091 | 1971.9339 | 0.867x |
| 8192x8192x8192 | True | 1.3247 | 2490.0863 | 1.0273 | 3210.9086 | 0.776x |
