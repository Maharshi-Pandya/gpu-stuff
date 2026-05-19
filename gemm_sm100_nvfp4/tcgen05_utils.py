import cutlass
import cutlass.cute as cute

from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cutlass_dsl import dsl_user_op


NVVM_CTA_GROUP_MAP = [None, nvvm.Tcgen05GroupKind.CTA_1, nvvm.Tcgen05GroupKind.CTA_2]


def _make_tmem_llvm_ptr(taddr, *, loc=None, ip=None):
    tmem_ptr = llvm.PointerType.get(cute.AddressSpace.tmem.value)
    return llvm.inttoptr(tmem_ptr, cutlass.Int32(taddr).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)



@dsl_user_op
def tma_bulk_g2s(
    smem_ptr: cute.Pointer, gmem_ptr: cute.Pointer,
    size: cutlass.Constexpr[int], mbar_ptr: cute.Pointer, *, loc=None, ip=None
):
    llvm.inline_asm(
        None,
        [
            smem_ptr.toint().ir_value(loc=loc, ip=ip),
            gmem_ptr.toint().ir_value(loc=loc, ip=ip),
            cutlass.Uint32(size).ir_value(loc=loc, ip=ip),
            mbar_ptr.toint().ir_value(loc=loc, ip=ip)
        ],
        f"cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        constraints="r,l,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip
    )


class Tcgen05:
    @dsl_user_op
    @staticmethod
    def get_taddr_from_ptr(tmem_ptr: cute.Pointer | int, *, loc=None, ip=None):
        int_ptr = tmem_ptr.toint(loc=loc, ip=ip) if isinstance(tmem_ptr, cute.Pointer) else tmem_ptr
        return _make_tmem_llvm_ptr(int_ptr, loc=loc, ip=ip)

    @dsl_user_op
    @staticmethod
    def mma_f16(
        tmem_ptr: cute.Pointer | int,
        adesc, bdesc,
        idesc,
        enable_input_d,
        cta_group: int = 1,
        *, loc=None, ip=None
    ):
        taddr = Tcgen05.get_taddr_from_ptr(tmem_ptr, loc=loc, ip=ip)
        nvvm.tcgen05_mma(
            nvvm.Tcgen05MMAKind.F16,
            NVVM_CTA_GROUP_MAP[cta_group],
            taddr,   # d
            cutlass.Uint64(adesc).ir_value(loc=loc, ip=ip),
            cutlass.Uint64(bdesc).ir_value(loc=loc, ip=ip), 
            cutlass.Int32(idesc).ir_value(loc=loc, ip=ip),
            cutlass.Boolean(enable_input_d).ir_value(loc=loc, ip=ip),
            loc=loc, ip=ip
        )

    @dsl_user_op
    @staticmethod
    def mma_nvfp4(
        tmem_ptr: cute.Pointer | int,
        adesc, bdesc,
        idesc,
        sfa_tmem: int,
        sfb_tmem: int,
        enable_input_d,
        cta_group: int = 1,
        *, loc=None, ip=None
    ):
        taddr = Tcgen05.get_taddr_from_ptr(tmem_ptr, loc=loc, ip=ip)
        sfa_tmem_addr = Tcgen05.get_taddr_from_ptr(sfa_tmem, loc=loc, ip=ip)
        sfb_tmem_addr = Tcgen05.get_taddr_from_ptr(sfb_tmem, loc=loc, ip=ip)
        nvvm.tcgen05_mma_block_scale(
            nvvm.Tcgen05MMAKind.MXF4NVF4,
            NVVM_CTA_GROUP_MAP[cta_group],
            taddr,
            cutlass.Uint64(adesc).ir_value(loc=loc, ip=ip),
            cutlass.Uint64(bdesc).ir_value(loc=loc, ip=ip), 
            cutlass.Int32(idesc).ir_value(loc=loc, ip=ip),
            cutlass.Boolean(enable_input_d).ir_value(loc=loc, ip=ip),
            sfa_tmem_addr, sfb_tmem_addr,
            scale_vec_size=nvvm.Tcgen05MMAScaleVecSize.X4,
            loc=loc, ip=ip
        )

    @dsl_user_op
    @staticmethod
    def cp_32x128b(tmem_ptr, smem_desc, cta_group: int = 1, *, loc=None, ip=None):
        taddr = Tcgen05.get_taddr_from_ptr(tmem_ptr, loc=loc, ip=ip)
        nvvm.tcgen05_cp(
            nvvm.Tcgen05CpShape.SHAPE_32x128b,
            taddr,
            cutlass.Uint64(smem_desc).ir_value(loc=loc, ip=ip),
            cta_group=NVVM_CTA_GROUP_MAP[cta_group],
            multicast=nvvm.Tcgen05CpMulticast.WARPX4
        )

    @dsl_user_op
    @staticmethod
    def ld(tmem_ptr, shape: str, num: int, *, loc=None, ip=None):
        if shape == "32x32b":
            kind = nvvm.Tcgen05LdStShape.SHAPE_32X32B
            num_regs = num
        elif shape == "16x128b":
            kind = nvvm.Tcgen05LdStShape.SHAPE_16X128B
            num_regs = num * 2
        elif shape == "16x256b":
            kind = nvvm.Tcgen05LdStShape.SHAPE_16X256B
            num_regs = num * 4
        else:
            raise ValueError(f"Unknown LdSt shape for tcgen05.ld: {shape}")
        
        taddr = Tcgen05.get_taddr_from_ptr(tmem_ptr, loc=loc, ip=ip)

        if num_regs == 1:
            reg = nvvm.tcgen05_ld(
                cutlass.Int32.mlir_type, kind, num_regs, taddr,
                loc=loc, ip=ip
            )
            reg_f32 = llvm.bitcast(cutlass.Float32.mlir_type, reg, loc=loc, ip=ip)
            return cutlass.Float32(reg_f32)
        else:
            vec_i32 = ir.VectorType.get([num_regs], cutlass.Int32.mlir_type, loc=loc)
            vec_f32 = ir.VectorType.get([num_regs], cutlass.Float32.mlir_type, loc=loc)
            regs = nvvm.tcgen05_ld(vec_i32, kind, num, taddr, loc=loc, ip=ip)
            regs_f32 = llvm.bitcast(vec_f32, regs, loc=loc, ip=ip)
            return cute.TensorSSA(regs_f32, (num_regs,), cutlass.Float32)
        
    @dsl_user_op
    @staticmethod
    def commit(mbar: cute.Pointer, cta_mask=None, cta_group: int = 1, *, loc=None, ip=None):
        mbar_ptr = mbar.to_llvm_ptr(loc=loc, ip=ip)
        group = NVVM_CTA_GROUP_MAP[cta_group]
        if cutlass.const_expr(cta_mask is not None):
            nvvm.tcgen05_commit_arrive(
                mbar_ptr, multicast_mask=cta_mask.ir_value(loc=loc, ip=ip),
                group=group, loc=loc, ip=ip
            )
        else:
            nvvm.tcgen05_commit_arrive(mbar_ptr, group=group, loc=loc, ip=ip)

    @dsl_user_op
    @staticmethod
    def fence_after_thread_sync(*, loc=None, ip=None):
        nvvm.tcgen05_fence(
            nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC,
            loc=loc, ip=ip
        )
    
    @dsl_user_op
    @staticmethod
    def fence_before_thread_sync(*, loc=None, ip=None):
        nvvm.tcgen05_fence(
            nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC,
            loc=loc, ip=ip
        )
