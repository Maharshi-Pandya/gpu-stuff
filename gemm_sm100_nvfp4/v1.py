import torch
import cutlass
import cutlass.cute as cute

import cutlass.utils as utils
import cutlass.pipeline as pipeline

from typing import Optional
from cutlass._mlir.dialects import nvvm
from cutlass.cute.nvgpu import cpasync
from cuda.bindings.driver import CUstream

from tcgen05_utils import Tcgen05, tma_bulk_g2s
from tracer import TraceContext

from functools import cache


WARP_SIZE = 32


class NVFP4Sm100Gemm:
    def __init__(self):
        # hardcoded for now
        self.cta_tiler_mnk = (128, 128, 256)
        self.BM, self.BN, self.BK = self.cta_tiler_mnk

        self.a_dtype = cutlass.Uint8
        self.b_dtype = cutlass.Uint8
        self.sfa_dtype = cutlass.Float8E4M3FN
        self.sfb_dtype = cutlass.Float8E4M3FN
        self.c_dtype = cutlass.BFloat16
        # in bytes
        self.a_size = self.BM * self.BK // 2
        self.b_size = self.BN * self.BK // 2
        self.sfa_size = self.BM * self.BK // 16
        self.sfb_size = self.BN * self.BK // 16

        self.smem_size = utils.get_smem_capacity_in_bytes()
        self.stage_size = self.a_size + self.b_size + self.sfa_size + self.sfb_size
        self.num_stages = self.smem_size // self.stage_size

        self.epi_warps = (0, 1, 2, 3)
        self.tma_warp = 4
        self.mma_warp = 5
        self.num_threads = len(
            [*self.epi_warps, self.tma_warp, self.mma_warp]
        ) * WARP_SIZE
        self.align_bytes = 1024

    @cute.jit
    def setup_AB(
        self, tensor: cute.Tensor, BR: cutlass.Constexpr[int],
        BC: cutlass.Constexpr[int]
    ):
        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        smem_ab_layout = cute.make_layout(
            shape=(BR, BC, self.num_stages),
            stride=(BC, 1, BR * BC)
        )
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        smem_ab_layout = cute.make_composed_layout(swizzle_128B, 0, smem_ab_layout)
        
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            tma_op, tensor, smem_ab_layout, (BR, BC)
        )
        return tma_atom, tma_tensor, smem_ab_layout
    
    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,    # uint8 (M, K / 2)
        mB: cute.Tensor,    # uint8 (N, K / 2)
        mSfa: cute.Tensor,  # Layout: [M/128, K/16/4, 32, 4, 4]
        mSfb: cute.Tensor,  # Layout: [N/128, K/16/4, 32, 4, 4]
        mGlobalScale: cute.Tensor,   # 1x fp32
        mC: cute.Tensor,
        stream: CUstream,
        trace_ptr: Optional[cutlass.Int64] = None
    ):
        A_args = self.setup_AB(mA, self.BM, self.BK // 2)
        B_args = self.setup_AB(mB, self.BN, self.BK // 2)

        M, _ = mA.shape
        N, _ = mB.shape

        grid = (cute.ceil_div(M, self.BM), cute.ceil_div(N, self.BN), 1)
        self.kernel(
            A_args, B_args, mSfa, mSfb, mGlobalScale, mC, self.BM, self.BN, self.BK,
            self.sfa_size, self.sfb_size, self.stage_size, trace_ptr
        ).launch(
            grid=grid,
            block=(self.num_threads, 1, 1),
            stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        A_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        B_args: tuple[cute.CopyAtom, cute.Tensor, cute.ComposedLayout],
        mSfa: cute.Tensor,
        mSfb: cute.Tensor,
        mGlobalScale: cute.Tensor,
        mC: cute.Tensor,
        BM: cutlass.Constexpr[int],
        BN: cutlass.Constexpr[int],
        BK: cutlass.Constexpr[int],
        SFA_SIZE: cutlass.Constexpr[int],
        SFB_SIZE: cutlass.Constexpr[int],
        STAGE_SIZE: cutlass.Constexpr[int],
        trace_ptr: Optional[cutlass.Int64] = None
    ):
        tctx = TraceContext.create(trace_ptr)

        tid = cute.arch.thread_idx()[0]
        bidm, bidn, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // cute.arch.WARP_SIZE)

        a_tma_atom, a_tma_tensor, sA_layout = A_args
        b_tma_atom, b_tma_tensor, sB_layout = B_args

        smem = utils.SmemAllocator()

        sA = smem.allocate_tensor(self.a_dtype, sA_layout.outer, byte_alignment=128, swizzle=sA_layout.inner)
        sB = smem.allocate_tensor(self.b_dtype, sB_layout.outer, byte_alignment=128, swizzle=sB_layout.inner)        
        sSfa = smem.allocate_array(self.sfa_dtype, SFA_SIZE * self.num_stages, byte_alignment=128)
        sSfb = smem.allocate_array(self.sfb_dtype, SFB_SIZE * self.num_stages, byte_alignment=128)
        tma_empty_mbar = smem.allocate_array(cutlass.Int64, self.num_stages)
        tma_full_mbar = smem.allocate_array(cutlass.Int64, self.num_stages)
        mainloop_done_mbar = smem.allocate(cutlass.Int64, byte_alignment=8)
        tmem_holding_buf = smem.allocate(cutlass.Int32)

        tmem_alloc_barrier = pipeline.NamedBarrier(1, self.num_threads)
        tmem = utils.TmemAllocator(
            tmem_holding_buf,
            allocator_warp_id=0,    # let 1 epi warp alloc/dealloc
            barrier_for_retrieve=tmem_alloc_barrier,
            is_two_cta=False,
        )

        K = a_tma_tensor.shape[1] * 2
        rest_k = K // 16 // 4
        num_k_tiles = cute.ceil_div(K, BK)
        MMA_K = cutlass.const_expr(64)  # fixed for fp4 gemm

        if warp_id == 0:
            for i in cutlass.range_constexpr(self.num_stages):
                cute.arch.mbarrier_init(tma_empty_mbar + i, 1)
                cute.arch.mbarrier_init(tma_full_mbar + i, 1)
            cute.arch.mbarrier_init(mainloop_done_mbar, 1)
            cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(a_tma_atom)
            cpasync.prefetch_descriptor(b_tma_atom)

        tmem.allocate(num_columns=512)
        tmem.wait_for_alloc()   # hack for sync threads
        tmem_ptr = tmem.retrieve_ptr()

        # TMA warp
        if warp_id == self.tma_warp:
            gA = cute.local_tile(a_tma_tensor, (BM, BK // 2), (bidm, None))
            gB = cute.local_tile(b_tma_tensor, (BN, BK // 2), (bidn, None))

            gA_ = cute.group_modes(gA, 0, 2)
            gB_ = cute.group_modes(gB, 0, 2)
            sA_ = cute.group_modes(sA, 0, 2)
            sB_ = cute.group_modes(sB, 0, 2)

            tstage = 0
            ephase = 1  # no consumer at start of producer
            for iter_k in cutlass.range(num_k_tiles, unroll=1):
                tctx.b("mma_wait")
                cute.arch.mbarrier_wait(tma_empty_mbar + tstage, ephase)
                tctx.e("mma_wait")

                mbar = tma_full_mbar + tstage
                # cpasync copy from gmem to smem using PTX with mbar
                # ptrs for this stage of SMEM's and GMEM's Sfa and Sfb
                # SF layout: [M/128, K/16/4, 32, 4, 4]
                ssfa_ptr = sSfa + tstage * SFA_SIZE
                ssfb_ptr = sSfb + tstage * SFB_SIZE
                off_k = iter_k * BK
                # 512 columns -> 32x4x4
                gsfa_ptr = mSfa.iterator + (bidm * rest_k + off_k // (16 * 4)) * 512
                gsfb_ptr = mSfb.iterator + (bidn * rest_k + off_k // (16 * 4)) * 512

                tctx.b("tma_load")
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar, STAGE_SIZE)
                    tma_bulk_g2s(ssfa_ptr, gsfa_ptr, SFA_SIZE, mbar)
                    tma_bulk_g2s(ssfb_ptr, gsfb_ptr, SFB_SIZE, mbar)
                utils.block_copy(a_tma_atom, gA_[None, iter_k], sA_[None, tstage], tma_bar_ptr=mbar)
                utils.block_copy(b_tma_atom, gB_[None, iter_k], sB_[None, tstage], tma_bar_ptr=mbar)
                tctx.e("tma_load")

                # increase stage with wrapping
                tstage = (tstage + 1) % self.num_stages
                if tstage == 0:
                    ephase ^= 1
            
        # MMA warp
        if warp_id == self.mma_warp:
            tstage = 0
            fphase = 0

            # for AB, 128B swizzling, LBO is assumed to be 1
            # for SF, no swizzling, and SBO is 8x16B = 128B, 
            # no LBO since there isn't a 8x2 tile, only 8x1
            ab_SBO = cutlass.const_expr(8 * 128)
            sf_SBO = cutlass.const_expr(8 * 16)
            
            # first tcgen.cp move all sSfa and sSfb to TMEM
            # d-tmem ends at column BN, sfa starts at BN,
            # sfb starts at (BN + 16)th column
            sfa_tmem = BN
            sfb_tmem = sfa_tmem + 4 * (BK // MMA_K)
            
            # ab_dtype = E2M1
            idesc = cutlass.const_expr(
                (1 << 7) | (1 << 10) | (BN >> 3 << 17) | (BM >> 7 << 27)
            )
            ab_sdesc = cutlass.const_expr(
                (ab_SBO & 0x3FFFF) >> 4 << 32 | (1 << 46) | (2 << 61)
            )   # 128B swizzled
            sf_desc = cutlass.const_expr(
                (sf_SBO & 0x3FFFF) >> 4 << 32 | (1 << 46)
            )   # no swizzling

            for iter_k in cutlass.range(num_k_tiles, unroll=1):
                tctx.b("tma_wait")
                cute.arch.mbarrier_wait(tma_full_mbar + tstage, fphase)
                Tcgen05.fence_after_thread_sync()
                tctx.e("tma_wait")

                ssfa_ptr = sSfa + tstage * SFA_SIZE
                ssfb_ptr = sSfb + tstage * SFB_SIZE
                # move SFs to TMEM first before starting any MMA
                with cute.arch.elect_one():
                    for k in cutlass.range_constexpr(BK // MMA_K):
                        sfa_desc = sf_desc | ((ssfa_ptr + k * 512).toint() >> 4)
                        sfb_desc = sf_desc | ((ssfb_ptr + k * 512).toint() >> 4)
                        # 32x128b always does warpx4
                        Tcgen05.cp_32x128b(sfa_tmem + k*4, sfa_desc)
                        Tcgen05.cp_32x128b(sfb_tmem + k*4, sfb_desc)
                
                # tcgen05.cp -> tcgen05.mma is automatically ordered implicitly
                adesc = ab_sdesc | (sA[None, None, tstage].iterator.toint() >> 4)
                bdesc = ab_sdesc | (sB[None, None, tstage].iterator.toint() >> 4)

                tctx.b("mma")
                with cute.arch.elect_one():
                    for k in cutlass.range_constexpr(BK // MMA_K):
                        Tcgen05.mma_nvfp4(
                            tmem_ptr, adesc, bdesc, idesc, 
                            sfa_tmem + k*4, sfb_tmem + k*4,
                            iter_k > 0 or k > 0     # enable-input-d
                        )
                        adesc += (32 >> 4)
                        bdesc += (32 >> 4)
                    Tcgen05.commit(tma_empty_mbar + tstage)
                
                tctx.e("mma")
                tstage = (tstage + 1) % self.num_stages
                if tstage == 0:
                    fphase ^= 1

            with cute.arch.elect_one():
                Tcgen05.commit(mainloop_done_mbar)

        # Epilogue
        if warp_id in self.epi_warps:
            global_scale = mGlobalScale[0]
            VECSIZE = cutlass.const_expr(16)
            # [(1, 16), (M, N // 16)]
            gC = cute.zipped_divide(mC, tiler=(1, VECSIZE))

            c_dtype = mC.element_type
            copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), c_dtype, 
                num_bits_per_copy=VECSIZE * c_dtype.width,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE
            )
            tctx.b("mainloop_wait")
            if warp_id == 0:
                cute.arch.mbarrier_wait(mainloop_done_mbar, 0)
            cute.arch.barrier(barrier_id=1, number_of_threads=len(self.epi_warps) * cute.arch.WARP_SIZE)
            Tcgen05.fence_after_thread_sync()
            tctx.e("mainloop_wait")

            tctx.b("epilogue")
            for i in cutlass.range_constexpr(BN // VECSIZE):
                # tmem bits: 16-32 lane, 0-15 column
                addr = ((warp_id * 32) << 16) | (i * VECSIZE)
                regs = Tcgen05.ld(addr, "32x32b", VECSIZE)
                nvvm.tcgen05_wait(nvvm.Tcgen05WaitKind.LOAD)

                acc = cute.make_rmem_tensor(VECSIZE, dtype=self.c_dtype)
                regs = regs * global_scale
                acc.store(regs.to(mC.element_type))

                tCgC = gC[(0, None), (bidm * BM + tid, bidn * (BN // VECSIZE) + i)]
                cute.copy(copy_atom, acc, tCgC)
            tctx.e("epilogue")

            cute.arch.barrier(barrier_id=1, number_of_threads=len(self.epi_warps) * cute.arch.WARP_SIZE)
            tmem.free(tmem_ptr)

        tctx.flush()

    @cache
    @staticmethod
    def compile(has_trace_ptr: bool = False):
        M = cute.sym_int(divisibility=128)
        N = cute.sym_int(divisibility=128)
        Kby2 = cute.sym_int(divisibility=64)
        flattened_sfa = cute.sym_int(divisibility=128)
        flattened_sfb = cute.sym_int(divisibility=128)

        A = cute.runtime.make_fake_tensor(cutlass.Uint8, (M, Kby2), (Kby2, 1), assumed_align=128)
        B = cute.runtime.make_fake_tensor(cutlass.Uint8, (N, Kby2), (Kby2, 1), assumed_align=128)
        # Phyiscal layout should be (M/128, K/64, 32, 4, 4) flattened
        Sfa = cute.runtime.make_fake_tensor(cutlass.Float8E4M3FN, (flattened_sfa, ), (1, ), assumed_align=128)
        Sfb = cute.runtime.make_fake_tensor(cutlass.Float8E4M3FN, (flattened_sfb, ), (1, ), assumed_align=128)
        GlobalScale = cute.runtime.make_fake_tensor(cutlass.Float32, (1, ), (0, ), assumed_align=128)
        C = cute.runtime.make_fake_tensor(cutlass.BFloat16, (M, N), (N, 1), assumed_align=128)
        
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = NVFP4Sm100Gemm()
        trace_ptr = cutlass.Int64(0) if has_trace_ptr else None
        return cute.compile(
            kernel, A, B, Sfa, Sfb, GlobalScale, C, stream, 
            trace_ptr=trace_ptr,
            options="--enable-tvm-ffi"
        )


@torch.library.custom_op("fp4::nvfp4_gemm_cutedsl", mutates_args=())
def nvfp4_gemm_cutedsl(
    A: torch.Tensor, B: torch.Tensor,   # row major B
    Sfa: torch.Tensor, Sfb: torch.Tensor,
    GlobalScale: torch.Tensor
) -> torch.Tensor:
    C = A.new_empty(A.shape[0], B.shape[0], dtype=torch.bfloat16)
    NVFP4Sm100Gemm.compile()(A, B, Sfa, Sfb, GlobalScale, C)
    return C


@nvfp4_gemm_cutedsl.register_fake
def _(
    A: torch.Tensor, B: torch.Tensor,   # row major B
    Sfa: torch.Tensor, Sfb: torch.Tensor,
    GlobalScale: torch.Tensor,
) -> torch.Tensor:
    return A.new_empty(A.shape[0], B.shape[0], dtype=torch.bfloat16)


@torch.library.custom_op("fp4::nvfp4_gemm_cublas", mutates_args=())
def nvfp4_gemm_cublas(
    A: torch.Tensor, B: torch.Tensor,   # row major B
    Sfa: torch.Tensor, Sfb: torch.Tensor,
    Gs_A: torch.Tensor, Gs_B: torch.Tensor
) -> torch.Tensor:
    return torch._scaled_mm_v2(
        A.view(torch.float4_e2m1fn_x2) if A.dtype == torch.uint8 else A,
        B.view(torch.float4_e2m1fn_x2).T if B.dtype == torch.uint8 else B.T,
        [Sfa.flatten(), Gs_A],
        [2, 0], [1, 0],
        [Sfb.flatten(), Gs_B],
        [2, 0], [1, 0],
        None,
        torch.bfloat16
    )

@nvfp4_gemm_cublas.register_fake
def _(
    A: torch.Tensor, B: torch.Tensor,   # row major B
    Sfa: torch.Tensor, Sfb: torch.Tensor,
    Gs_A: torch.Tensor, Gs_B: torch.Tensor
):
    return A.new_empty(A.shape[0], B.shape[0], dtype=torch.bfloat16)
