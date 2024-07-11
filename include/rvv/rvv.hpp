

#include <riscv_vector.h>
#include <cstddef>
#include <functional>
#include <iostream>
#include <type_traits>
#include <unistd.h>
#include <concepts>

#if !defined(RVV_LEN)
#define RVV_LEN __riscv_v_fixed_vlen
#endif

namespace rvv_impl {

    template <typename T, typename... U>
    concept IsAnyOf = (std::same_as<T, U> || ...);
    
    template<typename T>
    concept SignedSIMD = IsAnyOf<T, int8_t, int16_t, int32_t, int64_t>;

    template<typename T>
    concept UnignedSIMD = IsAnyOf<T, uint8_t, uint16_t, uint32_t, uint64_t>;

    template<typename T>
    concept FloatingSIMD = IsAnyOf<T, _Float16, float, double>;

    // simd_impl_base implements functions where intrinsic's 
    // signature differs for signed,unsigned and floating types
    template <typename T>
    struct simd_impl_base{
        static_assert(false, "The provided type is not supported by the SIMD implementation");
    };

    template <SignedSIMD T>
    struct simd_impl_base<T>{

        inline static auto get(auto vec, auto index, size_t size)
        {
            return __riscv_vmv_x(__riscv_vslidedown(vec, index, size));
        }

        // Arithmetic Operations

        inline static auto add(auto x, auto y, size_t size)
        {
            return __riscv_vadd(x, y, size);
        }

        inline static auto sub(auto x, auto y, size_t size)
        {
            return __riscv_vsub(x, y, size);
        }

        inline static auto multiply(auto x, auto y, size_t size)
        {
            return __riscv_vmul(x, y, size);
        }

        inline static auto divide(auto x, auto y, size_t size)
        {
            return __riscv_vdiv(x, y, size);
        }

        inline static auto min(auto x, auto y, size_t size)
        {
            return __riscv_vmin(x, y, size);
        }

        inline static auto max(auto x, auto y, size_t size)
        {
            return __riscv_vmax(x, y, size);
        }

        // Reduction Operations

        template <typename U>
        inline static T reduce_sum(U x, size_t size)
        {
            return __riscv_vmv_x(__riscv_vredsum(x, U(), size));
        }

        inline static T reduce_min(auto x, size_t size)
        {
            return __riscv_vmv_x(__riscv_vredmin(x, x, size));
        }

        inline static T reduce_max(auto x, size_t size)
        {
            return __riscv_vmv_x(__riscv_vredmax(x, x, size));
        }

        // Comparison Operations

        inline static auto equal(auto x, auto y, size_t size)
        {
            return __riscv_vmseq(x, y, size);
        }
        inline static auto not_equal(auto x, auto y, size_t size)
        {
            return __riscv_vmsne(x, y, size);
        }
        inline static auto greater_than(auto x, auto y, size_t size)
        {
            return __riscv_vmsgt(x, y, size);
        }
        inline static auto greater_eq(auto x, auto y, size_t size)
        {
            return __riscv_vmsge(x, y, size);
        }
        inline static auto less_than(auto x, auto y, size_t size)
        {
            return __riscv_vmslt(x, y, size);
        }
        inline static auto less_eq(auto x, auto y, size_t size)
        {
            return __riscv_vmsle(x, y, size);
        }

    };

    template <UnignedSIMD T>
    struct simd_impl_base<T>{

        inline static auto get(auto vec, auto index, size_t size)
        {
            return __riscv_vmv_x(__riscv_vslidedown(vec, index, size));
        }

        // Arithmetic Operations

        inline static auto add(auto x, auto y, size_t size)
        {
            return __riscv_vadd(x, y, size);
        }

        inline static auto sub(auto x, auto y, size_t size)
        {
            return __riscv_vsub(x, y, size);
        }

        inline static auto divide(auto x, auto y, size_t size)
        {
            return __riscv_vdivu(x, y, size);
        }

        inline static auto multiply(auto x, auto y, size_t size)
        {
            return __riscv_vmul(x, y, size);
        }

        
        inline static auto min(auto x, auto y, size_t size)
        {
            return __riscv_vminu(x, y, size);
        }

        inline static auto max(auto x, auto y, size_t size)
        {
            return __riscv_vmaxu(x, y, size);
        }
        
        // Reduction Operations
        template <typename U>
        inline static T reduce_sum(U x, size_t size)
        {
            return __riscv_vmv_x(__riscv_vredsum(x, U(), size));
        }

        inline static T reduce_min(auto x, size_t size)
        {
            return __riscv_vmv_x(__riscv_vredumin(x, x, size));
        }

        inline static T reduce_max(auto x, size_t size)
        {
            return __riscv_vmv_x(__riscv_vredumax(x, x, size));
        }

        // Comparison Operations

        inline static auto equal(auto x, auto y, size_t size)
        {
            return __riscv_vmseq(x, y, size);
        }
        inline static auto not_equal(auto x, auto y, size_t size)
        {
            return __riscv_vmsne(x, y, size);
        }
        inline static auto greater_than(auto x, auto y, size_t size)
        {
            return __riscv_vmsgtu(x, y, size);
        }
        inline static auto greater_eq(auto x, auto y, size_t size)
        {
            return __riscv_vmsgeu(x, y, size);
        }
        inline static auto less_than(auto x, auto y, size_t size)
        {
            return __riscv_vmsltu(x, y, size);
        }
        inline static auto less_eq(auto x, auto y, size_t size)
        {
            return __riscv_vmsleu(x, y, size);
        }

    };

    template <FloatingSIMD T>
    struct simd_impl_base<T>{

        inline static auto get(auto vec, auto index, size_t size)
        {
            return __riscv_vfmv_f(__riscv_vslidedown(vec, index, size));
        }

        // Arithmetic Operations

        inline static auto add(auto x, auto y, size_t size)
        {
            return __riscv_vfadd(x, y, size);
        }

        inline static auto sub(auto x, auto y, size_t size)
        {
            return __riscv_vfsub(x, y, size);
        }

        inline static auto divide(auto x, auto y, size_t size)
        {
            return __riscv_vfdiv(x, y, size);
        }

        inline static auto multiply(auto x, auto y, size_t size)
        {
            return __riscv_vfmul(x, y, size);
        }

        inline static auto min(auto x, auto y, size_t size)
        {
            return __riscv_vfmin(x, y, size);
        }

        inline static auto max(auto x, auto y, size_t size)
        {
            return __riscv_vfmax(x, y, size);
        }

        // Reduction Operations

        template <typename U>
        inline static T reduce_sum(U x, size_t size)
        {
            return __riscv_vfmv_f(__riscv_vfredusum(x, U(), size));
        }

        inline static T reduce_min(auto x, size_t size)
        {
            return __riscv_vfmv_f(__riscv_vfredmin(x, x, size));
        }

        inline static T reduce_max(auto x, size_t size)
        {
            return __riscv_vfmv_f(__riscv_vfredmax(x, x, size));
        }

        // Comparison Operations

        inline static auto equal(auto x, auto y, size_t size)
        {
            return __riscv_vmfeq(x, y, size);
        }
        inline static auto not_equal(auto x, auto y, size_t size)
        {
            return __riscv_vmfne(x, y, size);
        }
        inline static auto greater_than(auto x, auto y, size_t size)
        {
            return __riscv_vmfgt(x, y, size);
        }
        inline static auto greater_eq(auto x, auto y, size_t size)
        {
            return __riscv_vmfge(x, y, size);
        }
        inline static auto less_than(auto x, auto y, size_t size)
        {
            return __riscv_vmflt(x, y, size);
        }
        inline static auto less_eq(auto x, auto y, size_t size)
        {
            return __riscv_vmfle(x, y, size);
        }

    };

    // simd_impl implements functions where intrinsic's 
    // signature is specific to the datatype 
    template <typename T>
    struct simd_impl
    {
        static_assert(false, "The provided type is not supported by the SIMD implementation");
    };

    static constexpr int max_vector_pack_size = RVV_LEN / 8;


    // ----------------------------------------------------------------------
    template <int T>
    struct simd_impl_
    {
    };

    template <>
    struct simd_impl_<1>
    {
        inline static auto all_true()
        {
            // vuint8m1_t a = __riscv_vmv_v_x_u8m1(0u, RVV_LEN/8);
            // return __riscv_vmseq(a, 0u, RVV_LEN/8);
            return __riscv_vmseq(vuint8m1_t{}, vuint8m1_t{}, RVV_LEN/8);
        }
        // inline static auto first_true()
        // {
        //     return svptrue_pat_b8(SV_VL1);
        // }
        // inline static auto next_true(auto curr_true)
        // {
        //     return svpnext_b8(svptrue_b8(), curr_true);
        // }
        inline static auto count(auto pred, size_t size)
        {
            return __riscv_vcpop(pred, size);
        }
    };

    template <>
    struct simd_impl_<2>
    {
        inline static auto all_true()
        {
            return __riscv_vmseq(vuint16m1_t{}, vuint16m1_t{}, RVV_LEN/16);
            // return svptrue_b16();
        }
        // inline static auto first_true()
        // {
        //     return svptrue_pat_b16(SV_VL1);
        // }
        // inline static auto next_true(auto curr_true)
        // {
        //     return svpnext_b16(svptrue_b16(), curr_true);
        // }
        inline static auto count(auto pred, size_t size)
        {
            return __riscv_vcpop(pred, size);
        }
    };

    template <>
    struct simd_impl_<4>
    {
        inline static auto all_true()
        {
            return __riscv_vmseq(vuint32m1_t{}, vuint32m1_t{}, RVV_LEN/32);
            // return svptrue_b32();
        }
        // inline static auto first_true()
        // {
        //     return svptrue_pat_b32(SV_VL1);
        // }
        // inline static auto next_true(auto curr_true)
        // {
        //     return svpnext_b32(svptrue_b32(), curr_true);
        // }
        inline static auto count(auto pred, size_t size)
        {
            return __riscv_vcpop(pred, size);
        }
    };

    template <>
    struct simd_impl_<8>
    {
        inline static auto all_true()
        {
            return __riscv_vmseq(vuint64m1_t{}, vuint64m1_t{}, RVV_LEN/64);
            // return svptrue_b64();
        }

        // inline static auto first_true()
        // {
        //     return svptrue_pat_b64(SV_VL1);
        // }
        // inline static auto next_true(auto curr_true)
        // {
        //     return svpnext_b64(svptrue_b64(), curr_true);
        // }
        inline static auto count(auto pred, size_t size)
        {
            return __riscv_vcpop(pred, size);
        }
    };

    
    // ----------------------------------------------------------------------
    template <>
    struct simd_impl<int8_t> : simd_impl_base<int8_t>
    {
        using value_t = int8_t;
        typedef vint8m1_t Vector __attribute__((riscv_rvv_vector_bits(RVV_LEN)));
        typedef vbool8_t Predicate __attribute__((riscv_rvv_vector_bits(RVV_LEN / 8)));
        static constexpr std::size_t size = max_vector_pack_size / sizeof(value_t);

        template <typename T>
        inline static Vector load(const T* ptr)
        {
            return __riscv_vle8_v_i8m1(ptr, size);
        }

        inline static const value_t iota_array[16] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        inline static const Vector index0123 = load(iota_array);

        template <typename T>
        inline static void store(Vector vec, T* ptr)
        {
            __riscv_vse8(ptr, vec, size);
        }

        inline static Vector set(Vector vec, size_t index, value_t val)
        {
            Predicate mask = __riscv_vmseq(index0123, index, size);
            return __riscv_vmerge(vec, val, mask, size);
        }


        inline static Vector set(Vector vec, Predicate index, value_t val)
        {
            return __riscv_vmerge_vxm_i8m1(vec, val, index, size);
        }
 
        inline static Vector fill(value_t val)
        {
             return __riscv_vmv_v_x_i8m1(val, size);
        }

        // inline static Vector index_series(value_t base, value_t step)
        // {
    
        //         Vector vbase = fill(base);
        //         Vector vid = __riscv_vreinterpret_v_u8m1_i8m1(
        //                             __riscv_vid_v_u8m1(size));
        //         return __riscv_vmadd(vid, step, vbase, size);
        // //     return svindex_s8(base, step);
        // }

        // inline static const Vector index0123 = index_series(value_t(0), value_t(1));
    };

    template <>
    struct simd_impl<uint8_t> : simd_impl_base<uint8_t>
    {
        using value_t = uint8_t;
        typedef vuint8m1_t Vector __attribute__((riscv_rvv_vector_bits(RVV_LEN)));
        typedef vbool8_t Predicate __attribute__((riscv_rvv_vector_bits(RVV_LEN / 8)));
        static constexpr std::size_t size = max_vector_pack_size / sizeof(value_t);

        // inline static Vector index_series(value_t base, value_t step)
        // {
        //     Vector vbase = fill(base);
        //     Vector vid = __riscv_vid_v_u8m1(size);
        //     return __riscv_vmadd(vid, step, vbase, size);
        // }

        // inline static const Vector index0123 = index_series(value_t(0), value_t(1));

        template <typename T>
        inline static Vector load(const T* ptr)
        {
            return __riscv_vle8_v_u8m1(ptr, size);
        }

        inline static const value_t iota_array[16] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        inline static const Vector index0123 = load(iota_array);

        template <typename T>
        inline static void store(Vector vec, T* ptr)
        {
            __riscv_vse8(ptr, vec, size);
        }

        inline static Vector set(Vector vec, size_t index, value_t val)
        {
            Predicate mask = __riscv_vmseq(index0123, index, size);
            return __riscv_vmerge(vec, val, mask, size);
        }

      inline static Vector fill(value_t val)
        {
             return __riscv_vmv_v_x_u8m1(val, size);
        }

    };

    template <>
    struct simd_impl<int16_t> : simd_impl_base<int16_t>
    {
        using value_t = int16_t;
        typedef vint16m1_t Vector __attribute__((riscv_rvv_vector_bits(RVV_LEN)));
        typedef vbool16_t Predicate __attribute__((riscv_rvv_vector_bits(RVV_LEN / 16)));
        static constexpr std::size_t size = max_vector_pack_size / sizeof(value_t);

        template <typename T>
        inline static Vector load(const T* ptr)
        {
            return __riscv_vle16_v_i16m1(ptr, size);
        }

        inline static const value_t iota_array[16] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        inline static const Vector index0123 = load(iota_array);

        template <typename T>
        inline static void store(Vector vec, T* ptr)
        {
            __riscv_vse16(ptr, vec, size);
        }

        inline static Vector set(Vector vec, size_t index, value_t val)
        {
            Predicate mask = __riscv_vmseq(index0123, index, size);
            return __riscv_vmerge(vec, val, mask, size);
        }

 
      inline static Vector fill(value_t val)
        {
             return __riscv_vmv_v_x_i16m1(val, size);
        }
        // inline static Vector index_series(value_t base, value_t step)
        // {
        //         Vector vbase = fill(base);
        //         Vector vid = __riscv_vreinterpret_v_u16m1_i16m1(
        //                             __riscv_vid_v_u16m1(size));
        //         return __riscv_vmadd(vid, step, vbase, size);
        // }
        // inline static const Vector index0123 = index_series(value_t(0), value_t(1));
    };

    template <>
    struct simd_impl<uint16_t> : simd_impl_base<uint16_t>
    {
        using value_t = uint16_t;
        typedef vuint16m1_t Vector __attribute__((riscv_rvv_vector_bits(RVV_LEN)));
        typedef vbool16_t Predicate __attribute__((riscv_rvv_vector_bits(RVV_LEN / 16)));
        static constexpr std::size_t size = max_vector_pack_size / sizeof(value_t);

        template <typename T>
        inline static Vector load(const T* ptr)
        {
            return __riscv_vle16_v_u16m1(ptr, size);
        }

        inline static const value_t iota_array[16] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        inline static const Vector index0123 = load(iota_array);

        template <typename T>
        inline static void store(Vector vec, T* ptr)
        {
            __riscv_vse16(ptr, vec, size);
        }

        inline static Vector set(Vector vec, size_t index, value_t val)
        {
            Predicate mask = __riscv_vmseq(index0123, index, size);
            return __riscv_vmerge(vec, val, mask, size);
        }

 
        inline static Vector fill(value_t val)
        {
             return __riscv_vmv_v_x_u16m1(val, size);
        }
        // inline static Vector index_series(value_t base, value_t step)
        // {
    
        //         Vector vbase = fill(base);
        //         Vector vid = __riscv_vid_v_u16m1(size);
        //         return __riscv_vmadd(vid, step, vbase, size);
        // }
        // inline static const Vector index0123 = index_series(value_t(0), value_t(1));
    };
    template <>
    struct simd_impl<int32_t> : simd_impl_base<int32_t>
    {
        using value_t = int32_t;
        typedef vint32m1_t Vector __attribute__((riscv_rvv_vector_bits(RVV_LEN)));
        typedef vbool32_t Predicate __attribute__((riscv_rvv_vector_bits(RVV_LEN / 32)));
        static constexpr std::size_t size = max_vector_pack_size / sizeof(value_t);

        template <typename T>
        inline static Vector load(const T* ptr)
        {
            return __riscv_vle32_v_i32m1(ptr, size);
        }

        inline static const value_t iota_array[16] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        inline static const Vector index0123 = load(iota_array);

        template <typename T>
        inline static void store(Vector vec, T* ptr)
        {
            __riscv_vse32(ptr, vec, size);
        }

        inline static Vector set(Vector vec, size_t index, value_t val)
        {
            Predicate mask = __riscv_vmseq(index0123, index, size);
            return __riscv_vmerge(vec, val, mask, size);
        }

        inline static Vector fill(value_t val)
        {
            return __riscv_vmv_v_x_i32m1(val, size);
        }
    };

    template <>
    struct simd_impl<uint32_t> : simd_impl_base<uint32_t>
    {
        using value_t = uint32_t;
        typedef vuint32m1_t Vector __attribute__((riscv_rvv_vector_bits(RVV_LEN)));
        typedef vbool32_t Predicate __attribute__((riscv_rvv_vector_bits(RVV_LEN / 32)));
        static constexpr std::size_t size = max_vector_pack_size / sizeof(value_t);

        template <typename T>
        inline static Vector load(const T* ptr)
        {
            return __riscv_vle32_v_u32m1(ptr, size);
        }

        inline static const value_t iota_array[16] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        inline static const Vector index0123 = load(iota_array);

        template <typename T>
        inline static void store(Vector vec, T* ptr)
        {
            __riscv_vse32(ptr, vec, size);
        }

        inline static Vector set(Vector vec, size_t index, value_t val)
        {
            Predicate mask = __riscv_vmseq(index0123, index, size);
            return __riscv_vmerge(vec, val, mask, size);
        }

        inline static Vector fill(value_t val)
        {
             return __riscv_vmv_v_x_u32m1(val, size);
        }

    };
    // template <>
    // struct simd_impl<int64_t>
    // {
    //     using T = int64_t;
    //     typedef svint64_t Vector __attribute__((arm_sve_vector_bits(SVE_LEN)));
    //     static constexpr std::size_t size = max_vector_pack_size / sizeof(T);

    //     inline static Vector set(Vector vec, Predicate index, T val)
    //     {
    //         return svdup_s64_m(vec, index, val);
    //     }
    //     inline static Vector fill(T val)
    //     {
    //         return svdup_s64(val);
    //     }
    //     inline static Vector index_series(T base, T step)
    //     {
    //         return svindex_s64(base, step);
    //     }
    //     inline static const Vector index0123 = svindex_s64(T(0), T(1));
    // };

    // template <>
    // struct simd_impl<uint64_t>
    // {
    //     using T = uint64_t;
    //     typedef svuint64_t Vector __attribute__((arm_sve_vector_bits(SVE_LEN)));
    //     static constexpr std::size_t size = max_vector_pack_size / sizeof(T);

    //     inline static Vector set(Vector vec, Predicate index, T val)
    //     {
    //         return svdup_u64_m(vec, index, val);
    //     }
    //     inline static Vector fill(T val)
    //     {
    //         return svdup_u64(val);
    //     }
    //     inline static Vector index_series(T base, T step)
    //     {
    //         return svindex_u64(base, step);
    //     }
    //     inline static const Vector index0123 = svindex_u64(T(0), T(1));
    // };

    // // ----------------------------------------------------------------------
    // template <>
    // struct simd_impl<float16_t>
    // {
    //     typedef svfloat16_t Vector
    //         __attribute__((arm_sve_vector_bits(SVE_LEN)));
    //     static constexpr std::size_t size =
    //         max_vector_pack_size / sizeof(float16_t);

    //     inline static Vector set(Vector vec, Predicate index, float16_t val)
    //     {
    //         return svdup_f16_m(vec, index, val);
    //     }
    //     inline static Vector fill(float16_t val)
    //     {
    //         return svdup_f16(val);
    //     }
    //     inline static const float16_t iota_array[32] = {0, 1, 2, 3, 4, 5, 6, 7,
    //         8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    //         25, 26, 27, 28, 29, 30, 31};
    //     inline static const Vector index0123 = svld1(
    //         sve_impl::simd_impl_<sizeof(float16_t)>::all_true(), iota_array);
    // };

    template <>
    struct simd_impl<float> : simd_impl_base<float>
    {
        using value_t = float;
        typedef vfloat32m1_t Vector __attribute__((riscv_rvv_vector_bits(RVV_LEN)));
        typedef vbool32_t Predicate __attribute__((riscv_rvv_vector_bits(RVV_LEN / 32)));
        static constexpr std::size_t size = max_vector_pack_size / sizeof(value_t);

        template <typename T>
        inline static Vector load(const T* ptr)
        {
            return __riscv_vle32_v_f32m1(ptr, size);
        }

        inline static const value_t iota_array[16] = {
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        inline static const Vector index0123 = load(iota_array);

        template <typename T>
        inline static void store(Vector vec, T* ptr)
        {
            __riscv_vse32(ptr, vec, size);
        }


        inline static Vector set(Vector vec, size_t index, value_t val)
        {
            Predicate mask = __riscv_vmfeq(index0123, value_t(index), size);
            return __riscv_vfmerge(vec, val, mask, size);
        }


        inline static Vector fill(value_t val)
        {
            return __riscv_vfmv_v_f_f32m1(val, size);
        }

    };

    // template <>
    // struct simd_impl<double>
    // {
    //     typedef svfloat64_t Vector
    //         __attribute__((arm_sve_vector_bits(SVE_LEN)));
    //     static constexpr std::size_t size =
    //         max_vector_pack_size / sizeof(double);

    //     inline static Vector set(Vector vec, Predicate index, double val)
    //     {
    //         return svdup_f64_m(vec, index, val);
    //     }
    //     inline static Vector fill(double val)
    //     {
    //         return svdup_f64(val);
    //     }
    //     inline static const double iota_array[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    //     inline static const Vector index0123 =
    //         svld1(sve_impl::simd_impl_<sizeof(double)>::all_true(), iota_array);
    // };
}    // namespace sve_impl







namespace rvv::experimental { inline namespace parallelism_v2 {
    namespace simd_abi {
        struct scalar
        {
        };
        struct rvv_abi
        {
        };

        template <typename T>
        inline constexpr int max_fixed_size = rvv_impl::max_vector_pack_size;

        template <typename T>
        using compatible = rvv_abi;

        template <typename T>
        using native = rvv_abi;

        template <typename T, size_t N>
        using fixed_size = rvv_abi;

//         // template <class T, size_t N>
//         // struct deduce { using type = sve_abi; };

//         // template <class T, size_t N> using deduce_t = typename deduce<T, N>::type;
    }    // namespace simd_abi

    struct element_aligned_tag
    {
    };
    struct vector_aligned_tag
    {
    };
    template <size_t>
    struct overaligned_tag
    {
    };
    inline constexpr element_aligned_tag element_aligned{};
    inline constexpr vector_aligned_tag vector_aligned{};
    template <size_t N>
    inline constexpr overaligned_tag<N> overaligned{};

    // ----------------------------------------------------------------------
    // traits [simd.traits]
    // ----------------------------------------------------------------------
    template <class T>
    struct is_abi_tag : std::is_same<T, simd_abi::rvv_abi>
    {
    };
    template <class T>
    inline constexpr bool is_abi_tag_v = is_abi_tag<T>::value;

    template <class T>
    struct is_simd;
    template <class T>
    inline constexpr bool is_simd_v = is_simd<T>::value;

    template <class T>
    struct is_simd_mask;
    template <class T>
    inline constexpr bool is_simd_mask_v = is_simd_mask<T>::value;

    template <class T>
    struct is_simd_flag_type : std::false_type
    {
    };
    template <>
    struct is_simd_flag_type<element_aligned_tag> : std::true_type
    {
    };
    template <>
    struct is_simd_flag_type<vector_aligned_tag> : std::true_type
    {
    };
    template <class T>
    inline constexpr bool is_simd_flag_type_v = is_simd_flag_type<T>::value;

    template <class T, class Abi = simd_abi::compatible<T>>
    struct simd_size
    {
        static inline constexpr size_t value = rvv_impl::simd_impl<T>::size;
    };
    template <class T, class Abi = simd_abi::compatible<T>>
    inline constexpr size_t simd_size_v = simd_size<T, Abi>::value;

    template <class T, class U = typename T::value_type>
    struct memory_alignment
    {
        static inline constexpr size_t value = rvv_impl::max_vector_pack_size;
    };
    template <class T, class U = typename T::value_type>
    inline constexpr size_t memory_alignment_v = memory_alignment<T, U>::value;

    // ----------------------------------------------------------------------
    // class template simd [simd.class]
    // ----------------------------------------------------------------------
    template <class T, class Abi = simd_abi::compatible<T>>
    class simd;
    template <class T>
    using native_simd = simd<T, simd_abi::native<T>>;
    template <class T, size_t N>
    using fixed_size_simd = simd<T, simd_abi::fixed_size<T, N>>;

    // ----------------------------------------------------------------------
    // class template simd_mask [simd.mask.class]
    // ----------------------------------------------------------------------
    template <class T, class Abi = simd_abi::compatible<T>>
    class simd_mask;
    template <class T>
    using native_simd_mask = simd_mask<T, simd_abi::native<T>>;

    template <class T>
    struct is_simd : std::false_type
    {
    };

    template <typename T, typename Abi>
    struct is_simd<simd<T, Abi>> : std::true_type
    {
    };

    template <class T>
    struct is_simd_mask : std::false_type
    {
    };

    template <typename T, typename Abi>
    struct is_simd_mask<simd_mask<T, Abi>> : std::true_type
    {
    };

    // class template simd
    template <typename T, typename Abi>
    class simd
    {
    private:
        using Impl = typename rvv_impl::simd_impl<T>;
        using Vector = typename Impl::Vector;
        using Predicate = typename Impl::Predicate;
        Vector vec;
        static inline constexpr int T_size = sizeof(T);
        // TODO: Remove all_true if unused
        // static inline const Predicate all_true =
        //     rvv_impl::simd_impl_<T_size>::all_true();

    public:
        using value_type = T;
        using abi_type = Abi;
        using mask_type = simd_mask<T, Abi>;

        static inline const Vector index0123 =
            Impl::index0123;

        static inline constexpr std::size_t size()
        {
            return Impl::size;
        }

        // ----------------------------------------------------------------------
        //  constructors
        // ----------------------------------------------------------------------
        inline simd(const simd&) = default;
        inline simd(simd&&) noexcept = default;
        inline simd& operator=(const simd&) = default;
        inline simd& operator=(simd&&) noexcept = default;

        template <typename U, typename Flag>
        inline simd(U* ptr, Flag)
        {
            static_assert(std::is_same_v<std::remove_cvref_t<U>, T>,
                "pointer should be same type as value_type");
            static_assert(is_simd_flag_type_v<Flag>,
                "use element_aligned or vector_aligned tag");
            vec = Impl::load(ptr);
            // vec = svld1(all_true, ptr);
        }

        inline simd(T val = {})
        {
            vec = Impl::fill(val);
        }

        inline simd(Vector v)
        {
            vec = v;
        }

        // ----------------------------------------------------------------------
        //  load and store
        // ----------------------------------------------------------------------
        template <typename U, typename Flag>
        inline void copy_from(const U* ptr, Flag)
        {
            static_assert(std::is_same_v<std::remove_cvref_t<U>, T>,
                "pointer should be same type as value_type");
            static_assert(is_simd_flag_type_v<Flag>,
                "use element_aligned or vector_aligned tag");
            vec = Impl::load(ptr);
            // vec = svld1(all_true, ptr);
        }

        template <typename U, typename Flag>
        inline void copy_to(U* ptr, Flag) const
        {
            static_assert(std::is_same_v<std::remove_cvref_t<U>, T>,
                "pointer should be same type as value_type");
            static_assert(is_simd_flag_type_v<Flag>,
                "use element_aligned or vector_aligned tag");
            Impl::store(vec, ptr);
            // svst1(all_true, ptr, vec);
        }

        // ----------------------------------------------------------------------
        //  get and set
        // ----------------------------------------------------------------------
        T get(int idx) const
        {
            // if (idx < 0 || idx > (int) size())
            //     return -1;
            return Impl::get(vec, idx, Impl::size);
        }

        T operator[](int idx) const
        {
            // if (idx < 0 || idx > (int) size())
            //     return -1;
            return Impl::get(vec, idx, Impl::size);
        }

        void set(int idx, T val)
        {
            // if (idx < 0 || idx > (int) size())
            //     return;
            vec = Impl::set(vec, idx, val);
        }

        // ----------------------------------------------------------------------
        // ostream overload
        // ----------------------------------------------------------------------
        friend std::ostream& operator<<(std::ostream& os, const simd& x)
        {
            using type_ = std::decay_t<decltype(x)>;
            using value_type_ = typename type_::value_type;
            using printable_type =
                std::conditional_t<std::is_integral_v<value_type_>,
                    std::conditional_t<std::is_unsigned_v<value_type_>,
                        uint32_t, int32_t>,
                    value_type_>;

            os << "( ";
            for (int i = 0; i < (int) x.size(); i++)
            {
                os << printable_type(x[i]) << ' ';
            }
            os << ")";
            return os;
        }

        // ----------------------------------------------------------------------
        //  Reduction
        // ----------------------------------------------------------------------
        // TODO: These don't need to be public
        inline auto reduce_sum() const
        {
            return Impl::reduce_sum(vec, Impl::size);
        }

        inline auto reduce_min() const
        {
            return Impl::reduce_min(vec, Impl::size);
        }

        inline auto reduce_max() const
        {
            return Impl::reduce_max(vec, Impl::size);
        }

//         // ----------------------------------------------------------------------
//         //  First and last elements
//         // ----------------------------------------------------------------------
//         inline auto first() const
//         {
//             return svlasta(svpfalse(), vec);
//         }

//         inline auto last() const
//         {
//             return svlastb(svpfalse(), vec);
//         }

        // ----------------------------------------------------------------------
        //  unary operators [simd.unary]
        // ----------------------------------------------------------------------
        inline simd& operator++()
        {
            vec = Impl::add(vec, static_cast<T>(1), Impl::size);
            return *this;
        }

        inline auto operator++(int)
        {
            auto vec_copy = *this;
            vec = Impl::add(vec, static_cast<T>(1), Impl::size);
            return vec_copy;
        }

        inline simd& operator--()
        {
            vec = Impl::sub(vec, static_cast<T>(1), Impl::size);
            return *this;
        }

        inline auto operator--(int)
        {
            auto vec_copy = *this;
            vec = Impl::sub(vec, static_cast<T>(1), Impl::size);
            return vec_copy;
        }

        inline simd operator+() const
        {
            return *this;
        }

        inline simd operator-() const
        {
            auto vec_copy = *this;
            vec_copy.vec = Impl::multiply(vec_copy.vec, static_cast<T>(-1), Impl::size);
            return vec_copy;
        }

        // ----------------------------------------------------------------------
        // binary operators [simd.binary]
        // ----------------------------------------------------------------------
        inline friend simd operator+(const simd& x, const simd& y)
        {
            return Impl::add(x.vec, y.vec, Impl::size);
        }

        inline friend simd operator-(const simd& x, const simd& y)
        {
            return Impl::sub(x.vec, y.vec, Impl::size);
        }

        inline friend simd operator*(const simd& x, const simd& y)
        {
            return Impl::multiply(x.vec, y.vec, Impl::size);
        }

        inline friend simd operator/(const simd& x, const simd& y)
        {
            return Impl::divide(x.vec, y.vec, Impl::size);
        }

        inline friend simd operator&(const simd& x, const simd& y)
        {
            static_assert(std::is_integral_v<T>,
                "operator& only works for integeral types");
            return __riscv_vand(x.vec, y.vec, Impl::size);
        }

        inline friend simd operator|(const simd& x, const simd& y)
        {
            static_assert(std::is_integral_v<T>,
                "operator| only works for integeral types");
            return __riscv_vor(x.vec, y.vec, Impl::size);
        }

        inline friend simd operator^(const simd& x, const simd& y)
        {
            static_assert(std::is_integral_v<T>,
                "operator^ only works for integeral types");
            return __riscv_xor(x.vec, y.vec, Impl::size);
        }

//         // friend simd operator<<(const simd& x, typename std::make_unsigned<T>::type
//         // n)
//         // {
//         //     return left_shift(x.vec, n);
//         // }

//         // friend std::enable_if_t<std::is_unsigned_v<T>, simd>
//         // friend simd operator>>(const simd& x, T n)
//         // {
//         //     return right_shift(x.vec, n);
//         // }

        // ----------------------------------------------------------------------
        // compound assignment [simd.cassign]
        // ----------------------------------------------------------------------
        inline friend simd& operator+=(simd& x, const simd& y)
        {
            x.vec = Impl::add(x.vec, y.vec, Impl::size);
            return x;
        }

        inline friend simd& operator-=(simd& x, const simd& y)
        {
            x.vec = Impl::sub(x.vec, y.vec, Impl::size);
            return x;
        }

        inline friend simd& operator*=(simd& x, const simd& y)
        {
            x.vec = Impl::multiply(x.vec, y.vec, Impl::size);
            return x;
        }

        inline friend simd& operator/=(simd& x, const simd& y)
        {
            x.vec = Impl::divide(x.vec, y.vec, Impl::size);
            return x;
        }

        inline friend simd operator&=(simd& x, const simd& y)
        {
            static_assert(std::is_integral_v<T>,
                "operator&= only works for integeral types");
            x.vec = __riscv_vand(x.vec, y.vec, Impl::size);
            return x;
        }

        inline friend simd operator|=(simd& x, const simd& y)
        {
            static_assert(std::is_integral_v<T>,
                "operator|= only works for integeral types");
            x.vec = __riscv_vor(x.vec, y.vec, Impl::size);
            return x;
        }

        inline friend simd operator^=(simd& x, const simd& y)
        {
            static_assert(std::is_integral_v<T>,
                "operator^= only works for integeral types");
            x.vec = __riscv_vxor(x.vec, y.vec, Impl::size);
            return x;
        }

//         // friend simd operator<<=(simd& x, typename std::make_unsigned<T>::type n)
//         // {
//         //     x.vec = left_shift(x.vec, n);
//         //     return x;
//         // }

//         // friend simd operator>>=(simd& x, T n)
//         // {
//         //     x.vec = right_shift(x.vec, n);
//         //     return x;
//         // }

        // ----------------------------------------------------------------------
        // compares [simd.comparison]
        // ----------------------------------------------------------------------
        inline friend mask_type operator==(const simd& x, const simd& y)
        {
            return Impl::equal(x.vec, y.vec, Impl::size);
        }

        inline friend mask_type operator!=(const simd& x, const simd& y)
        {
            return Impl::not_equal(x.vec, y.vec, Impl::size);
        }

        inline friend mask_type operator>=(const simd& x, const simd& y)
        {
            return Impl::greater_eq(x.vec, y.vec, Impl::size);
        }

        inline friend mask_type operator<=(const simd& x, const simd& y)
        {
            return Impl::less_eq(x.vec, y.vec, Impl::size);
        }

        inline friend mask_type operator>(const simd& x, const simd& y)
        {
            return Impl::greater_than(x.vec, y.vec, Impl::size);
        }

        inline friend mask_type operator<(const simd& x, const simd& y)
        {
            return Impl::less_than(x.vec, y.vec, Impl::size);
        }


//     private:
//         template <typename T_, typename Abi_>
//         inline friend simd<T_, Abi_> choose(const simd_mask<T_, Abi_>& msk,
//             const simd<T_, Abi_>& t, const simd<T_, Abi_>& f);

//         template <typename T_, typename Abi_>
//         inline friend void mask_assign(const simd_mask<T_, Abi_>& msk,
//             simd<T_, Abi_>& v, const simd<T_, Abi_>& val);

        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> min(
            const simd<T_, Abi_>& x, const simd<T_, Abi_>& y);

        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> max(
            const simd<T_, Abi_>& x, const simd<T_, Abi_>& y);

//         template <typename t_, typename abi_>
//         inline friend simd<t_, abi_> copysign(const simd<t_, abi_>& valSrc, const simd<t_, abi_>& signSrc);

        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> sqrt(const simd<T_, Abi_>& x);

        template <typename T_, typename Abi_>
        inline friend simd<T_, Abi_> abs(const simd<T_, Abi_>& x);

//         template <typename T_, typename Abi_>
//         inline friend simd<T_, Abi_> fma(const simd<T_, Abi_>& a,
//             const simd<T_, Abi_>& b, const simd<T_, Abi_>& z);

        template <typename T_, typename Abi_, typename Op>
        inline friend T_ reduce(const simd<T_, Abi_>& x, Op op);

//         template <typename T_, typename Abi_, typename Op>
//         inline friend simd<T_, Abi_> inclusive_scan(
//             const simd<T_, Abi_>& x, Op op);

//         template <typename T_, typename Abi_, typename Op>
//         inline friend simd<T_, Abi_> exclusive_scan(
//             const simd<T_, Abi_>& x, Op op, T_ init);

//         template <typename T_, typename Abi_>
//         inline friend simd<T_, Abi_> index_series(T_ base, T_ step);

//         template <typename T_, typename Abi_>
//         inline friend simd<T_, Abi_> compact(
//             const simd_mask<T_, Abi_>& msk, const simd<T_, Abi_>& v);

//         template <typename T_, typename Abi_>
//         inline friend simd<T_, Abi_> splice(const simd_mask<T_, Abi_>& msk,
//             const simd<T_, Abi_>& x, const simd<T_, Abi_>& y);

//         template <typename T_, typename Abi_>
//         inline friend simd<T_, Abi_> interleave_even(
//             const simd<T_, Abi_>& x, const simd<T_, Abi_>& y);

//         template <typename T_, typename Abi_>
//         inline friend simd<T_, Abi_> interleave_odd(
//             const simd<T_, Abi_>& x, const simd<T_, Abi_>& y);

//         template <typename T_, typename Abi_>
//         inline friend simd<T_, Abi_> select_even(
//             const simd<T_, Abi_>& x, const simd<T_, Abi_>& y);

//         template <typename T_, typename Abi_>
//         inline friend simd<T_, Abi_> select_odd(
//             const simd<T_, Abi_>& x, const simd<T_, Abi_>& y);

//         template <typename T_, typename Abi_>
//         inline friend simd<T_, Abi_> lower_half(
//             const simd<T_, Abi_>& x, const simd<T_, Abi_>& y);

//         template <typename T_, typename Abi_>
//         inline friend simd<T_, Abi_> upper_half(
//             const simd<T_, Abi_>& x, const simd<T_, Abi_>& y);

//         template <typename T_, typename Abi_>
//         inline friend simd<T_, Abi_> reverse(const simd<T_, Abi_>& x);
    }; //class simd

    template <typename T, typename Abi>
    inline simd<T, Abi> min(const simd<T, Abi>& x, const simd<T, Abi>& y)
    {
        return simd<T, Abi>::Impl::min(x.vec, y.vec, x.size());
    }

    template <typename T, typename Abi>
    inline simd<T, Abi> max(const simd<T, Abi>& x, const simd<T, Abi>& y)
    {
        return simd<T, Abi>::Impl::max(x.vec, y.vec, x.size());
    }

    template <typename T, typename Abi>
    inline std::pair<simd<T, Abi>, simd<T, Abi>> minmax(
        const simd<T, Abi>& x, const simd<T, Abi>& y)
    {
        return {min(x, y), max(x, y)};
    }

//     template <typename T_, typename Abi_>
//     inline simd<T_, Abi_> copysign(const simd<T_, Abi_>& valSrc, const simd<T_, Abi_>& signSrc) {
//         static_assert(
//             std::is_same_v<T_, float64_t>,
//             "vectorized copysign is only implemented for f64 types thus far.");
//         // obtain signbits by using a static -0.0 mask to find them in the signSrc via a bitwise AND
//         const auto signmask = sve_impl::simd_impl<T_>::fill(-0.0);
//         const auto signbits = svand_x(signSrc.all_true, svreinterpret_s64(signSrc.vec), svreinterpret_s64(signmask));
//         // obtains the valuebits by using the inverted signmask via a bitwise ANDNOT
//         const auto absbits = svbic_x(signSrc.all_true, svreinterpret_s64(valSrc.vec), svreinterpret_s64(signmask));
//         // Results is just the two bitsequeences combined via a bitwise OR
//         const auto result = svorr_x(signSrc.all_true, signbits, absbits);
//         // Return bit sequence as a f64 type
//         return svreinterpret_f64(result);
//     }

    template <typename T_, typename Abi_>
    inline simd<T_, Abi_> sqrt(const simd<T_, Abi_>& x)
    {
        static_assert(std::is_floating_point_v<T_>, "sqrt only works for floating point types");
        return __riscv_vfsqrt(x.vec, x.size());
    }

    template <typename T_, typename Abi_>
    inline simd<T_, Abi_> abs(const simd<T_, Abi_>& x)
    {
        if constexpr (std::is_floating_point_v<T_>)
        {
            return __riscv_vfabs(x.vec, x.size());
        }
        else if constexpr (std::is_signed_v<T_>)
        {
            using simd_vec_t = simd<T_, Abi_>::Vector;
            auto lt_zero_mask = __riscv_vmslt(x.vec, 0, x.size()); 
            return __riscv_vneg(lt_zero_mask, x.vec, x.size()); 
        } 
        else
        {
            return x;
        }
    }

//     template <typename T_, typename Abi_>
//     inline simd<T_, Abi_> fma(const simd<T_, Abi_>& a, const simd<T_, Abi_>& b,
//         const simd<T_, Abi_>& z)
//     {
//         return svmad_m(a.all_true, a.vec, b.vec, z.vec);
//     }

    template <typename T, typename Abi, typename Op = std::plus<>>
    inline T reduce(const simd<T, Abi>& x, Op op = {})
    {
        if constexpr (std::is_same_v<Op, std::plus<>>)
        {
            return x.reduce_sum();
        }
        else
        {
            // Need unsigned type of same size as T, for index vector in gather operation
            using unsigned_T = std::conditional_t<sizeof(T) == 1, uint8_t,
                            std::conditional_t<sizeof(T) == 2, uint16_t,
                            std::conditional_t<sizeof(T) == 4, uint32_t, 
                            std::conditional_t<sizeof(T) == 8, uint64_t, void>>>>;

            static_assert(!std::is_same_v<unsigned_T, void>, "Unsupported type for reduce operation");

            using unsigned_simd_t = simd<unsigned_T, Abi>;
            // Create indices [0,0,1,1,2,2...size/2 - 1] for gather operation
            unsigned_simd_t dup_ind_low = (unsigned_simd_t::index0123)/2;
            // Create indices [size/2, size/2, ... size-1]
            unsigned_simd_t dup_ind_high = dup_ind_low + unsigned_T(unsigned_simd_t::size())/2;

            using simd_t = simd<T, Abi>;
            auto x_vec = x.vec;

            for (std::size_t i = 1; i < simd_t::size(); i *= 2)
            {
                x_vec =
                    op(simd_t(__riscv_vrgather(x_vec, dup_ind_low.vec, simd_t::size())), 
                    simd_t(__riscv_vrgather(x_vec, dup_ind_high.vec, simd_t::size()))
                    ).vec;
            }

            return simd_t(x_vec).get(0);
        }
    }



//     template <typename T_, typename Abi_, typename Op = std::plus<>>
//     inline simd<T_, Abi_> inclusive_scan(const simd<T_, Abi_>& x, Op op = {})
//     {
//         using simd_t = simd<T_, Abi_>;
//         auto x_vec = x.vec;
//         auto iota_vec = sve_impl::simd_impl<T_>::index0123;
//         auto local_all_true = x.all_true;
//         if constexpr (simd_t::size() >= 2)
//         {
//             x_vec = svsplice(svcmplt(local_all_true, iota_vec, T_(1)), x_vec,
//                 op(simd_t(x_vec), simd_t(svext(x_vec, x_vec, 1))).vec);
//         }
//         if constexpr (simd_t::size() >= 4)
//         {
//             x_vec = svsplice(svcmplt(local_all_true, iota_vec, T_(2)), x_vec,
//                 op(simd_t(x_vec), simd_t(svext(x_vec, x_vec, 2))).vec);
//         }
//         if constexpr (simd_t::size() >= 8)
//         {
//             x_vec = svsplice(svcmplt(local_all_true, iota_vec, T_(4)), x_vec,
//                 op(simd_t(x_vec), simd_t(svext(x_vec, x_vec, 4))).vec);
//         }
//         if constexpr (simd_t::size() >= 16)
//         {
//             x_vec = svsplice(svcmplt(local_all_true, iota_vec, T_(8)), x_vec,
//                 op(simd_t(x_vec), simd_t(svext(x_vec, x_vec, 8))).vec);
//         }
//         if constexpr (simd_t::size() >= 32)
//         {
//             x_vec = svsplice(svcmplt(local_all_true, iota_vec, T_(16)), x_vec,
//                 op(simd_t(x_vec), simd_t(svext(x_vec, x_vec, 16))).vec);
//         }
//         if constexpr (simd_t::size() == 64)
//         {
//             x_vec = svsplice(svcmplt(local_all_true, iota_vec, T_(32)), x_vec,
//                 op(simd_t(x_vec), simd_t(svext(x_vec, x_vec, 32))).vec);
//         }
//         return x_vec;
//     }

//     template <typename T_, typename Abi_, typename Op = std::plus<>>
//     inline simd<T_, Abi_> exclusive_scan(
//         const simd<T_, Abi_>& x, Op op = {}, T_ init = {})
//     {
//         auto x_vec = x.vec;
//         x_vec = svinsr(x.vec, init);
//         return inclusive_scan(simd<T_, Abi_>(x_vec));
//     }

//     template <typename T_, typename Abi_ = simd_abi::sve_abi>
//     inline simd<T_, Abi_> index_series(T_ base, T_ step)
//     {
//         if constexpr (std::is_floating_point_v<T_> ||
//             std::is_same_v<T_, float16_t>)
//         {
//             return exclusive_scan(simd<T_, Abi_>(step)) + base;
//         }
//         else
//         {
//             return sve_impl::simd_impl<T_>::index_series(base, step);
//         }
//     }

//     template <typename T_, typename Abi_>
//     inline simd<T_, Abi_> interleave_even(
//         const simd<T_, Abi_>& x, const simd<T_, Abi_>& y)
//     {
//         return svtrn1(x.vec, y.vec);
//     }

//     template <typename T_, typename Abi_>
//     inline simd<T_, Abi_> interleave_odd(
//         const simd<T_, Abi_>& x, const simd<T_, Abi_>& y)
//     {
//         return svtrn2(x.vec, y.vec);
//     }

//     template <typename T_, typename Abi_>
//     inline simd<T_, Abi_> select_even(
//         const simd<T_, Abi_>& x, const simd<T_, Abi_>& y)
//     {
//         return svuzp1(x.vec, y.vec);
//     }

//     template <typename T_, typename Abi_>
//     inline simd<T_, Abi_> select_odd(
//         const simd<T_, Abi_>& x, const simd<T_, Abi_>& y)
//     {
//         return svuzp2(x.vec, y.vec);
//     }

//     template <typename T_, typename Abi_>
//     inline simd<T_, Abi_> lower_half(
//         const simd<T_, Abi_>& x, const simd<T_, Abi_>& y)
//     {
//         return svzip1(x.vec, y.vec);
//     }

//     template <typename T_, typename Abi_>
//     inline simd<T_, Abi_> upper_half(
//         const simd<T_, Abi_>& x, const simd<T_, Abi_>& y)
//     {
//         return svzip2(x.vec, y.vec);
//     }

//     template <typename T_, typename Abi_>
//     inline simd<T_, Abi_> reverse(const simd<T_, Abi_>& x)
//     {
//         return svrev(x.vec);
//     }

    template <typename T, typename Abi>
    class simd_mask
    {
    private:
        using Predicate = typename rvv_impl::simd_impl<T>::Predicate;
        static inline constexpr int T_size = sizeof(T);
        Predicate pred;
        static inline const Predicate all_true =
            rvv_impl::simd_impl_<T_size>::all_true();

    public:
        using value_type = bool;
        using simd_type = simd<T, Abi>;
        using abi_type = Abi;

        static inline const auto index0123 = simd_type::index0123;
        static inline constexpr std::size_t size()
        {
            return simd<T, Abi>::size();
        }

        // ----------------------------------------------------------------------
        //  constructors
        // ----------------------------------------------------------------------
        inline simd_mask(const simd_mask&) = default;
        inline simd_mask(simd_mask&&) = default;
        inline simd_mask& operator=(const simd_mask&) = default;
        inline simd_mask& operator=(simd_mask&&) = default;

        inline simd_mask(bool val = false)
        {
            if (val)
            {
                pred = all_true;
            }
            else
                pred = __riscv_vmnot(all_true, size());
        }

        inline simd_mask(Predicate p)
        {
            pred = p;
        }

        // ----------------------------------------------------------------------
        //  get and set
        // ----------------------------------------------------------------------
        bool get(int idx) const
        {
            if (idx < 0 || idx > (int) size())
                throw std::out_of_range("index out of range");

            auto index_mask = (index0123==simd_type(T(idx)));
            return rvv_impl::simd_impl_<T_size>::count(
                __riscv_vmand(pred, index_mask.pred, size()), size());
        }

        bool operator[](int idx) const
        {
            return get(idx);
        }

        void set(int idx, bool val)
        {
            if (idx < 0 || idx > (int) size())
                throw std::out_of_range("index out of range");

            auto index_mask = (index0123==simd_type(T(idx)));
            if (val)
                pred = __riscv_vmor(pred, index_mask.pred, size());
            else
                pred = __riscv_vmandn(pred, index_mask.pred, size());
        }

        // ----------------------------------------------------------------------
        // ostream overload
        // ----------------------------------------------------------------------
        friend std::ostream& operator<<(std::ostream& os, const simd_mask& x)
        {
            using type_ = std::decay_t<decltype(x)>;
            using simd_type_ = typename type_::simd_type;
            using value_type_ = typename simd_type::value_type;

            os << "( ";
            for (int i = 0; i < (int) x.size(); i++)
            {
                os << x[i] << ' ';
            }
            os << ")";
            return os;
        }

        // ----------------------------------------------------------------------
        //  unary operators
        // ----------------------------------------------------------------------
        inline simd_mask operator!() const noexcept
        {
            return __riscv_vmnot(pred, size());
        }

        // ----------------------------------------------------------------------
        //  binary operators
        // ----------------------------------------------------------------------
        inline friend simd_mask operator&&(
            const simd_mask& x, const simd_mask& y) noexcept
        {
            return __riscv_vmand(x.pred, y.pred, size());
        }

        inline friend simd_mask operator||(
            const simd_mask& x, const simd_mask& y) noexcept
        {
            return __riscv_vmor(x.pred, y.pred, size());
        }

        inline friend simd_mask operator&(
            const simd_mask& x, const simd_mask& y) noexcept
        {
            return __riscv_vmand(x.pred, y.pred, size());
        }

        inline friend simd_mask operator|(
            const simd_mask& x, const simd_mask& y) noexcept
        {
            return __riscv_vmor(x.pred, y.pred, size());
        }

        inline friend simd_mask operator^(
            const simd_mask& x, const simd_mask& y) noexcept
        {
            return __riscv_vmxor(x.pred, y.pred, size());
        }

        // ----------------------------------------------------------------------
        // simd_mask compound assignment [simd.mask.cassign]
        // ----------------------------------------------------------------------
        inline friend simd_mask& operator&=(
            simd_mask& x, const simd_mask& y) noexcept
        {
            x.pred = __riscv_vmand(x.pred, y.pred, size());
            return x;
        }

        inline friend simd_mask& operator|=(
            simd_mask& x, const simd_mask& y) noexcept
        {
            x.pred = __riscv_vmor(x.pred, y.pred, size());
            return x;
        }

        inline friend simd_mask& operator^=(
            simd_mask& x, const simd_mask& y) noexcept
        {
            x.pred = __riscv_vmxor(x.pred, y.pred, size());
            return x;
        }

        // ----------------------------------------------------------------------
        // simd_mask compares [simd.mask.comparison]
        // ----------------------------------------------------------------------
        inline friend simd_mask operator==(
            const simd_mask& x, const simd_mask& y) noexcept
        {
            return __riscv_vmnor(x.pred, y.pred, size());
        }

        inline friend simd_mask operator!=(
            const simd_mask& x, const simd_mask& y) noexcept
        {
            return __riscv_vmor(x.pred, y.pred, size());
        }

        // ----------------------------------------------------------------------
        //  algorithms
        // ----------------------------------------------------------------------
        inline int popcount() const
        {
            return rvv_impl::simd_impl_<T_size>::count(pred, size());
        }

        inline bool all_of() const
        {
            return popcount() == (int) size();
        }

        inline bool any_of() const
        {
            return popcount() > 0;
        }

        inline bool none_of() const
        {
            return popcount() != (int) size();
        }

        inline bool some_of() const
        {
            int c = popcount();
            return (c > 0) && (c < (int) size());
        }

//         inline int find_first_set() const
//         {
//             auto index = sve_impl::simd_impl_<T_size>::first_true();
//             for (int i = 0; i < (int) size(); i++)
//             {
//                 if (sve_impl::simd_impl_<T_size>::count(
//                         svand_z(all_true, pred, index)))
//                     return i;
//                 index = sve_impl::simd_impl_<T_size>::next_true(index);
//             }
//             return -1;
//         }

//         inline int find_last_set() const
//         {
//             int ans = -1;
//             auto index = sve_impl::simd_impl_<T_size>::first_true();
//             for (int i = 0; i < (int) size(); i++)
//             {
//                 if (sve_impl::simd_impl_<T_size>::count(
//                         svand_z(all_true, pred, index)))
//                     ans = i;
//                 index = sve_impl::simd_impl_<T_size>::next_true(index);
//             }
//             return ans;
//         }

//     private:
//         template <typename T_, typename Abi_>
//         inline friend simd<T_, Abi_> choose(const simd_mask<T_, Abi_>& msk,
//             const simd<T_, Abi_>& t, const simd<T_, Abi_>& f);

//         template <typename T_, typename Abi_>
//         inline friend void mask_assign(const simd_mask<T_, Abi_>& msk,
//             simd<T_, Abi_>& v, const simd<T_, Abi_>& val);

//         template <typename T_, typename Abi_>
//         inline friend simd<T_, Abi_> compact(
//             const simd_mask<T_, Abi_>& msk, const simd<T_, Abi_>& v);

//         template <typename T_, typename Abi_>
//         inline friend simd<T_, Abi_> splice(const simd_mask<T_, Abi_>& msk,
//             const simd<T_, Abi_>& x, const simd<T_, Abi_>& y);
    }; //class simd_mask

//     template <class T, class Abi>
//     inline bool all_of(const simd_mask<T, Abi>& m)
//     {
//         return m.all_of();
//     }

//     template <class T, class Abi>
//     inline bool any_of(const simd_mask<T, Abi>& m)
//     {
//         return m.any_of();
//     }

//     template <class T, class Abi>
//     inline bool none_of(const simd_mask<T, Abi>& m)
//     {
//         return m.none_of();
//     }

//     template <class T, class Abi>
//     inline bool some_of(const simd_mask<T, Abi>& m)
//     {
//         return m.some_of();
//     }

//     template <class T, class Abi>
//     inline int popcount(const simd_mask<T, Abi>& m)
//     {
//         return m.popcount();
//     }

//     template <class T, class Abi>
//     inline int find_first_set(const simd_mask<T, Abi>& m)
//     {
//         return m.find_first_set();
//     }

//     template <class T, class Abi>
//     inline int find_last_set(const simd_mask<T, Abi>& m)
//     {
//         return m.find_last_set();
//     }

//     template <typename T, typename Abi>
//     inline simd<T, Abi> choose(const simd_mask<T, Abi>& msk,
//         const simd<T, Abi>& t, const simd<T, Abi>& f)
//     {
//         return svsel(msk.pred, t.vec, f.vec);
//     }

//     template <typename T, typename Abi>
//     inline void mask_assign(
//         const simd_mask<T, Abi>& msk, simd<T, Abi>& v, const simd<T, Abi>& val)
//     {
//         v.vec = svsel(msk.pred, val.vec, v.vec);
//     }

//     template <typename T, typename Abi>
//     inline simd<T, Abi> compact(
//         const simd_mask<T, Abi>& msk, const simd<T, Abi>& v)
//     {
//         static_assert(sizeof(T) >= 4,
//             "compact function only works with airthmetic types which\
//          are atleast 4 bytes in size");
//         return svcompact(msk.pred, v.vec);
//     }

//     template <typename T_, typename Abi_>
//     inline simd<T_, Abi_> splice(const simd_mask<T_, Abi_>& msk,
//         const simd<T_, Abi_>& x, const simd<T_, Abi_>& y)
//     {
//         return svsplice(msk.pred, x.vec, y.vec);
//     }
}}    // namespace rvv::experimental::parallelism_v2
