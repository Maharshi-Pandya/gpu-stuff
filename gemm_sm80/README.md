## Results

Tested on an **NVIDIA GeForce RTX 4060 Laptop GPU** (Ada Lovelace, SM 8.9, 8GB VRAM).

| | Time (ms) | TFLOPS | Efficiency |
|---|---|---|---|
| cuBLAS (via `torch.mm`) | 4.555 | 30.17 | 100% |
| **CuTeDSL kernel v1** | **5.894** | **23.32** | **77.3%** |

Problem size: `M=N=K=4096`, dtype `bfloat16`, computing `C = A @ B.T`.

Correctness: **max error = 0.0000**, exact match with cuBLAS output.
