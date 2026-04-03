"""
SM80 GEMM Kernel using CUTLASS Python DSL
==========================================

Computes C = A @ B.T where:
  - A is (M, K) row-major (K contiguous)
  - B is (N, K) row-major (K contiguous) — stored like PyTorch nn.Linear weights
  - C is (M, N) row-major (N contiguous)

Tiling hierarchy:
  Global → Threadblock tile (BLOCK_M × BLOCK_N, accumulated over BLOCK_K chunks)
    → Warp tile (WARP_M × WARP_N)
      → MMA tile (16 × 8, via mma.m16n8k16 tensor core instruction)

Each threadblock has NUM_WARP_M × NUM_WARP_N warps (default 2×2 = 4 warps = 128 threads).
Each warp performs NUM_MMA_M × NUM_MMA_N MMA operations per k-step.

Key PTX instructions used:
  - cp.async.cg.shared.global: async copy from global to shared memory (no register detour)
  - ldmatrix.sync.aligned: warp-cooperative load from shared memory to registers
  - mma.sync.aligned.m16n8k16: tensor core matrix multiply-accumulate
  - cvt.rn.bf16x2.f32: convert two FP32 values to packed BF16 pair
"""

import torch
import cutlass
import cutlass.cute as cute

from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
import cuda.bindings.driver as cuda


# MMA instruction shape: m16n8k16
# Each mma.sync computes a (16×16) @ (16×8) = (16×8) tile
MMA_M = 16
MMA_N = 8
MMA_K = 16
WARP_SIZE = 32

torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.int32: cutlass.Int32,
    torch.int64: cutlass.Int64,
}


# =============================================================================
# PTX Helper Operations (inline assembly wrappers)
# =============================================================================

@dsl_user_op
def get_ptr_as_int64(
    tensor: cute.Tensor, offset: cutlass.Int32, *, loc=None, ip=None
) -> cutlass.Int64:
    """Extract a 64-bit global memory address from a CuTe tensor.
    Used for global memory pointers (A, B, C) which need 64-bit addressing."""
    elem_ptr = tensor.iterator + cutlass.Int32(offset)
    ptr_int = llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip)
    return cutlass.Int64(ptr_int)


@dsl_user_op
def get_ptr_as_uint32(
    tensor: cute.Tensor, offset: cutlass.Int32, *, loc=None, ip=None
) -> cutlass.Uint32:
    """Extract a 32-bit shared memory address from a CuTe tensor.
    Shared memory uses 32-bit addressing in PTX (generic address space)."""
    elem_ptr = tensor.iterator + cutlass.Int32(offset)
    ptr_int = llvm.ptrtoint(T.i32(), elem_ptr.llvm_ptr, loc=loc, ip=ip)
    return cutlass.Uint32(ptr_int)


@dsl_user_op
def ldmatrix_x2(
    base_ptr: cutlass.Uint32, *, loc=None, ip=None
) -> tuple[cutlass.Uint32, cutlass.Uint32]:
    """Load two 8×8 matrices from shared memory using warp-cooperative load.

    ldmatrix is a warp-level operation: all 32 threads participate simultaneously.
    - Threads 0-7 provide addresses for the 1st 8×8 matrix
    - Threads 8-15 provide addresses for the 2nd 8×8 matrix
    - Threads 16-31 are unused but must still participate

    Each thread gets back 2 registers (2 × uint32 = 4 × bf16 values).
    Used for loading B operand fragments (8×16 = two 8×8 blocks).

    IMPORTANT: has_side_effects=True prevents the compiler from reordering
    this load before __syncthreads() or eliminating "redundant" calls.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [cutlass.Uint32(base_ptr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {$0, $1}, [$2];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip
    )
    v0 = cutlass.Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip))
    v1 = cutlass.Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip))
    return v0, v1


@dsl_user_op
def ldmatrix_x4(
    base_ptr: cutlass.Uint32, *, loc=None, ip=None
) -> tuple[cutlass.Uint32, cutlass.Uint32, cutlass.Uint32, cutlass.Uint32]:
    """Load four 8×8 matrices from shared memory using warp-cooperative load.

    Thread-to-matrix mapping for 16×16 A operand:
      [8×8-0] [8×8-2]    threads 0-7 → rows 0-7 of block 0
      [8×8-1] [8×8-3]    threads 8-15 → rows 0-7 of block 1
                          threads 16-23 → rows 0-7 of block 2
                          threads 24-31 → rows 0-7 of block 3

    Each thread gets 4 registers (4 × uint32 = 8 × bf16 values).
    Used for loading A operand fragments (16×16 = four 8×8 blocks).
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [cutlass.Uint32(base_ptr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip
    )
    v0 = cutlass.Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip))
    v1 = cutlass.Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip))
    v2 = cutlass.Uint32(llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip))
    v3 = cutlass.Uint32(llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip))
    return v0, v1, v2, v3


@dsl_user_op
def cp_async_16(
    dst_smem_uint32: cutlass.Uint32,
    src_global_int64: cutlass.Int64,
    *, loc=None, ip=None
) -> cutlass.Int32:
    """Asynchronous copy of 16 bytes from global memory to shared memory.

    Bypasses registers entirely (data goes directly from L2 cache to shared memory).
    Must be followed by cp_async_commit_group() and cp_async_wait_group()
    before the data can be read from shared memory.

    16 bytes = 8 bf16 elements per copy.
    """
    llvm.inline_asm(
        None,
        [
            cutlass.Uint32(dst_smem_uint32).ir_value(loc=loc, ip=ip),
            cutlass.Int64(src_global_int64).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mma_m16n8k16(
    a0: cutlass.Uint32, a1: cutlass.Uint32, a2: cutlass.Uint32, a3: cutlass.Uint32,
    b0: cutlass.Uint32, b1: cutlass.Uint32,
    c0: cutlass.Float32, c1: cutlass.Float32, c2: cutlass.Float32, c3: cutlass.Float32,
    *, loc=None, ip=None
) -> tuple[cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32]:
    """Tensor core matrix multiply-accumulate: D = A * B + C

    Computes (16×16) @ (16×8) = (16×8) using bf16 inputs and fp32 accumulation.

    Per-thread register usage:
      A: 4 uint32 registers (= 8 bf16 values, distributed across 32 threads)
      B: 2 uint32 registers (= 4 bf16 values, distributed across 32 threads)
      C/D: 4 float32 registers (= 4 fp32 values, distributed across 32 threads)

    The "row.col" layout means:
      - A is treated as row-major (K contiguous) — matches our sA layout
      - B is treated as col-major (K contiguous) — our sB is (N,K) row-major,
        which is identical to (K,N) col-major in memory

    Constraint "0,1,2,3" ties C inputs to D outputs (same physical registers),
    matching the CUDA "+f" read-write constraint. This is required because the
    hardware instruction reads C and writes D in the same register slots.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            cutlass.Uint32(a0).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(a1).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(a2).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(a3).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(b0).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(b1).ir_value(loc=loc, ip=ip),
            cutlass.Float32(c0).ir_value(loc=loc, ip=ip),
            cutlass.Float32(c1).ir_value(loc=loc, ip=ip),
            cutlass.Float32(c2).ir_value(loc=loc, ip=ip),
            cutlass.Float32(c3).ir_value(loc=loc, ip=ip),
        ],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, "       # D (output accumulator)
        "{$4, $5, $6, $7}, "       # A (4 registers)
        "{$8, $9}, "               # B (2 registers)
        "{$10, $11, $12, $13};",   # C (input accumulator, tied to D)
        "=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip
    )
    v0 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip))
    v1 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip))
    v2 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip))
    v3 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip))
    return v0, v1, v2, v3


@dsl_user_op
def cvt_f32x2_to_bf16x2(
    v0: cutlass.Float32, v1: cutlass.Float32, *, loc=None, ip=None
) -> cutlass.Uint32:
    """Convert two FP32 values to a packed BF16×2 uint32.

    Used in the epilogue to convert FP32 accumulators back to BF16 before
    storing to global memory. The "$2, $1" operand order controls packing:
    v0 goes into the lower 16 bits, v1 into the upper 16 bits.
    """
    result = llvm.inline_asm(
        T.i32(),
        [cutlass.Float32(v0).ir_value(loc=loc, ip=ip), cutlass.Float32(v1).ir_value(loc=loc, ip=ip)],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc, ip=ip
    )
    return cutlass.Uint32(result)


@dsl_user_op
def store_global_u32(
    ptr: cutlass.Int64, value: cutlass.Uint32, *, loc=None, ip=None
):
    """Store a 32-bit value (packed bf16×2) to global memory.
    Each store writes 2 bf16 output elements at once."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(ptr).ir_value(loc=loc, ip=ip),
            cutlass.Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# =============================================================================
# Global → Shared Memory Copy
# =============================================================================

@cute.jit
def global_to_shared_async(
    in_ptr: cutlass.Int64, in_stride: cutlass.Uint32,
    out_ptr: cutlass.Uint32, tid: cutlass.Uint32,
    TB_SIZE: cutlass.Constexpr[int], HEIGHT: cutlass.Constexpr[int],
    WIDTH: cutlass.Constexpr[int], OUT_STRIDE: cutlass.Constexpr[int],
    elem_bytes: cutlass.Constexpr[int], num_elems: cutlass.Constexpr[int]
):
    """Cooperatively copy a (HEIGHT × WIDTH) tile from global to shared memory.

    All threads in the threadblock participate. Each thread copies 16 bytes
    (num_elems elements) per iteration. The loop is fully unrolled at compile time.

    For BLOCK_M=128, BLOCK_K=64, TB_SIZE=128, num_elems=8:
      total_elements = 128 × 64 = 8192
      elements_per_iter = 128 × 8 = 1024
      num_iters = 8192 / 1024 = 8

    in_ptr/out_ptr are byte addresses; all offsets are scaled by elem_bytes.
    """
    num_iters = cutlass.const_expr((HEIGHT * WIDTH) // (TB_SIZE * num_elems))
    for iter in cutlass.range_constexpr(num_iters):
        idx = (iter * TB_SIZE + tid) * num_elems
        row = idx // WIDTH
        col = idx % WIDTH

        dst_addr = out_ptr + (row * OUT_STRIDE + col) * elem_bytes
        cp_async_16(dst_addr, in_ptr + (row * in_stride + col) * elem_bytes)


# =============================================================================
# GEMM Kernel
# =============================================================================

class GemmSM80Kernel:
    def __init__(
            self, ab_dtype: cutlass.Numeric, c_dtype: cutlass.Numeric,
            BLOCK_M: int = 128, BLOCK_N: int = 128, BLOCK_K: int = 64,
            NUM_WARP_M: int = 2, NUM_WARP_N: int = 2
        ):
        self.ab_dtype = ab_dtype
        self.c_dtype = c_dtype

        assert BLOCK_M % NUM_WARP_M == 0
        assert BLOCK_N % NUM_WARP_N == 0
        assert BLOCK_K % MMA_K == 0

        # Threadblock tile shape
        self.BLOCK_M = BLOCK_M  # 128 rows of A
        self.BLOCK_N = BLOCK_N  # 128 rows of B (= columns of output)
        self.BLOCK_K = BLOCK_K  # 64 columns processed per mainloop iteration

        # Warp grid: how the threadblock tile is divided among warps
        self.NUM_WARP_M = NUM_WARP_M  # 2 warps along M
        self.NUM_WARP_N = NUM_WARP_N  # 2 warps along N

        # Per-warp tile shape
        self.WARP_M = self.BLOCK_M // self.NUM_WARP_M  # 64 rows per warp
        self.WARP_N = self.BLOCK_N // self.NUM_WARP_N  # 64 cols per warp
        assert self.WARP_M % MMA_M == 0
        assert self.WARP_N % MMA_N == 0

        # Number of MMA operations each warp performs per k-step
        self.NUM_MMA_M = self.WARP_M // MMA_M  # 64/16 = 4 MMAs along M
        self.NUM_MMA_N = self.WARP_N // MMA_N   # 64/8  = 8 MMAs along N

        # Total threads per threadblock
        self.TB_SIZE = self.NUM_WARP_M * self.NUM_WARP_N * WARP_SIZE  # 2×2×32 = 128

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        stream: cuda.CUstream
    ):
        # Per-thread register counts for one MMA operation
        num_acc_regs = (MMA_M * MMA_N) // WARP_SIZE                                 # 16*8/32 = 4 fp32 regs
        num_A_regs = (MMA_M * MMA_K * (self.ab_dtype.width // 8)) // 4 // WARP_SIZE  # 16*16*2/4/32 = 4 uint32 regs
        num_B_regs = (MMA_N * MMA_K * (self.ab_dtype.width // 8)) // 4 // WARP_SIZE   # 8*16*2/4/32 = 2 uint32 regs

        M, K = mA.shape
        N, _ = mB.shape

        # 1D grid: each block handles one (BLOCK_M × BLOCK_N) output tile
        grid_size = cute.ceil_div(M, self.BLOCK_M) * cute.ceil_div(N, self.BLOCK_N)
        self.kernel(
            mA, mB, mC,
            self.BLOCK_M, self.BLOCK_N, self.BLOCK_K,
            self.NUM_WARP_M, self.NUM_WARP_N, self.WARP_M,
            self.WARP_N, self.NUM_MMA_M, self.NUM_MMA_N, self.TB_SIZE,
            num_acc_regs, num_A_regs, num_B_regs
        ).launch(
            grid=[grid_size, 1, 1],
            block=[self.TB_SIZE, 1, 1],
            stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        BLOCK_M: cutlass.Constexpr[int],
        BLOCK_N: cutlass.Constexpr[int],
        BLOCK_K: cutlass.Constexpr[int],
        NUM_WARP_M: cutlass.Constexpr[int],
        NUM_WARP_N: cutlass.Constexpr[int],
        WARP_M: cutlass.Constexpr[int],
        WARP_N: cutlass.Constexpr[int],
        NUM_MMA_M: cutlass.Constexpr[int],
        NUM_MMA_N: cutlass.Constexpr[int],
        TB_SIZE: cutlass.Constexpr[int],
        num_acc_regs: cutlass.Constexpr[int],
        num_A_regs: cutlass.Constexpr[int],
        num_B_regs: cutlass.Constexpr[int]
    ):
        # ── Thread / block identification ──
        tid = cute.arch.thread_idx()[0]
        bid = cute.arch.block_idx()[0]
        warp_id = cute.arch.warp_idx()    # tid // 32
        lane_id = cute.arch.lane_idx()    # tid % 32

        M, K = mA.shape
        N, _ = mB.shape

        # Shared memory stride (no padding or swizzle in this version)
        SHM_STRIDE = BLOCK_K

        # ── Threadblock tile assignment ──
        # Map 1D block index to 2D tile coordinates (row-major block ordering)
        num_blocks_n = cute.ceil_div(N, BLOCK_N)
        bid_m = bid // num_blocks_n
        bid_n = bid % num_blocks_n
        offset_m = bid_m * BLOCK_M   # starting row in A/C
        offset_n = bid_n * BLOCK_N   # starting row in B / column in C

        # ── Warp tile assignment within the threadblock ──
        # 2×2 warp grid: warp_id_m ∈ {0,1}, warp_id_n ∈ {0,1}
        warp_id_m = warp_id // NUM_WARP_N
        warp_id_n = warp_id % NUM_WARP_N

        # ── Setup global memory byte-pointers ──
        # All pointer arithmetic is in bytes (scaled by elem_bytes)
        A = get_ptr_as_int64(mA, 0)
        B = get_ptr_as_int64(mB, 0)
        C = get_ptr_as_int64(mC, 0)
        ab_elem_bytes = cutlass.const_expr(self.ab_dtype.width // 8)  # 2 for bf16
        c_elem_bytes = cutlass.const_expr(mC.element_type.width // 8)  # 2 for bf16

        # Advance A to this block's row chunk: A[offset_m, 0]
        A += offset_m * K * ab_elem_bytes
        # Advance B to this block's row chunk: B[offset_n, 0]
        B += offset_n * K * ab_elem_bytes
        # Advance C to this warp's output tile: C[offset_m + warp_m_offset, offset_n + warp_n_offset]
        C += ((offset_m + warp_id_m * WARP_M) * N + (offset_n + warp_id_n * WARP_N)) * c_elem_bytes

        # ── Shared memory allocation ──
        # sA: (BLOCK_M × BLOCK_K) for A tile, row-major
        # sB: (BLOCK_N × BLOCK_K) for B tile, row-major
        # Total: (128 + 128) × 64 × 2 = 32KB shared memory
        smem = cutlass.utils.SmemAllocator()
        sA_layout = cute.make_ordered_layout((BLOCK_M, BLOCK_K), order=(1, 0))
        sB_layout = cute.make_ordered_layout((BLOCK_N, BLOCK_K), order=(1, 0))
        sA = smem.allocate_tensor(mA.element_type, sA_layout, byte_alignment=16)
        sB = smem.allocate_tensor(mB.element_type, sB_layout, byte_alignment=16)

        # Get shared memory base addresses as 32-bit integers for PTX instructions
        A_shared = get_ptr_as_uint32(sA, 0)
        B_shared = get_ptr_as_uint32(sB, 0)

        # ── Accumulator registers ──
        # Each thread holds NUM_MMA_M × NUM_MMA_N × 4 fp32 accumulators
        # = 4 × 8 × 4 = 128 float registers per thread
        num_elems = cutlass.const_expr(16 // ab_elem_bytes)  # 8 bf16 elements per 16-byte copy
        acc_layout = cute.make_ordered_layout(
            (NUM_MMA_M, NUM_MMA_N, num_acc_regs), order=(2, 1, 0)
        )
        acc = cute.make_rmem_tensor(acc_layout, cutlass.Float32)
        acc.fill(0.0)  # zero-initialize accumulators (D = A*B + 0 on first iteration)

        # =================================================================
        # MAINLOOP: iterate over K dimension in chunks of BLOCK_K
        # =================================================================
        for block_k in range(0, K, BLOCK_K):

            # ── Stage 1: Global → Shared memory (async copy) ──
            # All 128 threads cooperatively load the A and B tiles
            global_to_shared_async(
                A, K, A_shared, tid,
                TB_SIZE, BLOCK_M, BLOCK_K, SHM_STRIDE, ab_elem_bytes, num_elems
            )
            global_to_shared_async(
                B, K, B_shared, tid,
                TB_SIZE, BLOCK_N, BLOCK_K, SHM_STRIDE, ab_elem_bytes, num_elems
            )

            # Commit and wait for all async copies to complete
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()  # ensure all threads see the loaded data

            # ── Stage 2: Shared memory → Registers → MMA ──
            # Process BLOCK_K in steps of MMA_K=16
            for k in cutlass.range_constexpr(0, BLOCK_K, MMA_K):

                # Compute shared memory offsets for this warp's tile at k-offset
                A_shm_warp = A_shared + ((warp_id_m * WARP_M) * SHM_STRIDE + k) * ab_elem_bytes
                B_shm_warp = B_shared + ((warp_id_n * WARP_N) * SHM_STRIDE + k) * ab_elem_bytes

                # ── Load B fragments ──
                # Load all NUM_MMA_N=8 B fragments first (reused across all M tiles)
                # Each fragment is 8×16 (NUM_MMA_N × MMA_K), loaded as two 8×8 sub-matrices
                b_layout = cute.make_ordered_layout((NUM_MMA_N, num_B_regs), order=(1, 0))
                B_reg = cute.make_rmem_tensor(b_layout, dtype=cute.Uint32)

                for n in cutlass.range_constexpr(0, NUM_MMA_N):
                    # ldmatrix addressing for B:
                    #   row = n * MMA_N + (lane_id % 8)    → which of the 8 rows in this 8×8 block
                    #   col = (lane_id // 8) * 8            → which 8×8 sub-block (0 or 8)
                    B_ptr = B_shm_warp + (
                        (n * MMA_N + (lane_id % 8)) * SHM_STRIDE + (lane_id // 8) * 8
                    ) * ab_elem_bytes
                    B_reg[n, 0], B_reg[n, 1] = ldmatrix_x2(B_ptr)

                # ── Load A fragments and compute MMAs ──
                for m in cutlass.range_constexpr(0, NUM_MMA_M):
                    # Load one 16×16 A fragment using ldmatrix.x4
                    A_reg = cute.make_rmem_tensor((num_A_regs,), dtype=cutlass.Uint32)

                    # ldmatrix addressing for A:
                    #   row = m * MMA_M + (lane_id % 16)   → which of 16 rows
                    #   col = (lane_id // 16) * 8           → which 8-column group (0 or 8)
                    A_ptr = A_shm_warp + (
                        (m * MMA_M + (lane_id % 16)) * SHM_STRIDE + (lane_id // 16) * 8
                    ) * ab_elem_bytes
                    A_reg[0], A_reg[1], A_reg[2], A_reg[3] = ldmatrix_x4(A_ptr)

                    # Multiply this A fragment with all B fragments
                    for n in cutlass.range_constexpr(0, NUM_MMA_N):
                        acc[m, n, 0], acc[m, n, 1], acc[m, n, 2], acc[m, n, 3] = mma_m16n8k16(
                            A_reg[0], A_reg[1], A_reg[2], A_reg[3],
                            B_reg[n, 0], B_reg[n, 1],
                            acc[m, n, 0], acc[m, n, 1], acc[m, n, 2], acc[m, n, 3]
                        )

            # Sync before next iteration overwrites shared memory
            cute.arch.sync_threads()

            # Advance global pointers to next BLOCK_K chunk
            A += BLOCK_K * ab_elem_bytes
            B += BLOCK_K * ab_elem_bytes

        # =================================================================
        # EPILOGUE: Convert FP32 accumulators to BF16 and store to C
        # =================================================================
        # MMA output layout (per thread, per m16n8k16 tile):
        #   acc[0], acc[1] → C[row,     col], C[row,     col+1]
        #   acc[2], acc[3] → C[row + 8, col], C[row + 8, col+1]
        # where:
        #   row = lane_id / 4       (0..7 for top half, same for bottom half via +8)
        #   col = (lane_id % 4) * 2 (0, 2, 4, 6)
        #
        # Each thread stores 2 bf16 values per store (packed into uint32).
        # Total stores per thread: NUM_MMA_M × NUM_MMA_N × 2 = 4 × 8 × 2 = 64

        for m in cutlass.range_constexpr(0, NUM_MMA_M):
            for n in cutlass.range_constexpr(0, NUM_MMA_N):
                row = m * MMA_M + (lane_id // 4)
                col = n * MMA_N + (lane_id % 4) * 2

                # Store top half (rows 0-7 of this MMA tile)
                store_global_u32(
                    C + ((row + 0) * N + col) * c_elem_bytes,
                    cvt_f32x2_to_bf16x2(acc[m, n, 0], acc[m, n, 1])
                )
                # Store bottom half (rows 8-15 of this MMA tile)
                store_global_u32(
                    C + ((row + 8) * N + col) * c_elem_bytes,
                    cvt_f32x2_to_bf16x2(acc[m, n, 2], acc[m, n, 3])
                )


# =============================================================================
# Host-side compilation and dispatch
# =============================================================================

def _gemm_sm80(
    A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
):
    """Compile (once) and run the GEMM kernel."""
    ab_dtype, c_dtype = [
        torch2cute_dtype_map[t.dtype] if t is not None else None
        for t in [A, C]
    ]

    compile_key = (ab_dtype, c_dtype)
    if compile_key not in _gemm_sm80.compile_cache:
        # Create symbolic tensors for compilation (shapes are dynamic)
        m, n, k = cute.sym_int(), cute.sym_int(), cute.sym_int()
        A_cute = cute.runtime.make_fake_compact_tensor(
            ab_dtype, (m, k), stride_order=(1, 0), assumed_align=128
        )
        B_cute = cute.runtime.make_fake_compact_tensor(
            ab_dtype, (n, k), stride_order=(1, 0), assumed_align=128
        )
        C_cute = cute.runtime.make_fake_compact_tensor(
            ab_dtype, (m, n), stride_order=(1, 0), assumed_align=128
        )
        _gemm_sm80.compile_cache[compile_key] = cute.compile(
            GemmSM80Kernel(ab_dtype, c_dtype),
            A_cute, B_cute, C_cute,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi"
        )

    _gemm_sm80.compile_cache[compile_key](A, B, C)

_gemm_sm80.compile_cache = {}


def gemm(A: torch.Tensor, B: torch.Tensor):
    """Compute C = A @ B.T using the custom SM80 GEMM kernel."""
    M, _ = A.shape
    N, _ = B.shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    _gemm_sm80(A, B, C)
    return C


# =============================================================================
# Main: correctness check + benchmark
# =============================================================================

if __name__ == "__main__":
    M, N, K = 4096, 4096, 4096
    a = torch.randn((M, K), dtype=torch.bfloat16, device="cuda:0")
    b = torch.randn((N, K), dtype=torch.bfloat16, device="cuda:0")

    c_new = gemm(a, b)
    c_ref = a @ b.T

    print(f"\n>> A: {a.shape}\n{a}")
    print(f"\n>> B: {b.shape}\n{b}")
    print(f"\n>> C_ref (cuBLAS):\n{c_ref}")
    print(f"\n>> C_new (custom):\n{c_new}")

    # Correctness
    max_err = (c_ref - c_new).abs().max().item()
    mean_err = (c_ref - c_new).abs().mean().item()
    allclose = torch.allclose(c_ref, c_new, atol=1.0, rtol=0.01)
    print(f"\nMax error: {max_err:.4f} | Mean error: {mean_err:.4f} | allclose: {allclose}")

    # Benchmark
    from triton.testing import do_bench

    custom_ms = do_bench(lambda: gemm(a, b), warmup=100, rep=500)
    ref_ms = do_bench(lambda: torch.mm(a, b.T), warmup=100, rep=500)

    flops = 2.0 * M * N * K
    ref_tflops = flops / (ref_ms * 1e-3) / 1e12
    custom_tflops = flops / (custom_ms * 1e-3) / 1e12
    pct = (custom_tflops / ref_tflops) * 100

    print(f"\ncuBLAS:        {ref_ms:.3f} ms | {ref_tflops:.2f} TFLOPS")
    print(f"Custom kernel: {custom_ms:.3f} ms | {custom_tflops:.2f} TFLOPS")
    print(f"Efficiency:    {pct:.1f}% of cuBLAS\n")
