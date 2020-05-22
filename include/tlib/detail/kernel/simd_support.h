#ifndef TLIB_DETAIL_KERNEL_SIMD_SUPPORT_H
#define TLIB_DETAIL_KERNEL_SIMD_SUPPORT_H

#include <immintrin.h>
namespace tlib::simd::detail{

    struct simd_config{
        #ifdef __SSE__
            static constexpr bool sse = true;
        #else
            static constexpr bool sse = false;
        #endif

        #ifdef __SSE2__
            static constexpr bool sse2 = true;
        #else
            static constexpr bool sse2 = false;
        #endif

        #ifdef __SSE3__
            static constexpr bool sse3 = true;
        #else
            static constexpr bool sse3 = false;
        #endif

        #ifdef __SSE4_1__
            static constexpr bool sse4_1 = true;
        #else
            static constexpr bool sse4_1 = false;
        #endif

        #ifdef __AVX__
            static constexpr bool avx = true;
        #else
            static constexpr bool avx = false;
        #endif

        #ifdef __AVX2__
            static constexpr bool avx2 = true;
        #else
            static constexpr bool avx2 = false;
        #endif

        #ifdef __AVX512F__
            static constexpr bool avx512f = true;
        #else
            static constexpr bool avx512f = false;
        #endif

        #ifdef __AVX512BW__
            static constexpr bool avx512bw = true;
        #else
            static constexpr bool avx512bw = false;
        #endif

        #ifdef __AVX512CD__
            static constexpr bool avx512cd = true;
        #else
            static constexpr bool avx512cd = false;
        #endif

        #ifdef __AVX512DQ__
            static constexpr bool avx512dq = true;
        #else
            static constexpr bool avx512dq = false;
        #endif

        #ifdef __AVX512VL__
            static constexpr bool avx512vl = true;
        #else
            static constexpr bool avx512vl = false;
        #endif

        #if defined(__FMA__)
            static constexpr bool fma = true;
        #else
            static constexpr bool fma = false;
        #endif

    };

    template<bool B,typename>
    struct wrap_cond{
        static constexpr bool value = B;
    };

    template<bool B,typename T>
    inline static constexpr auto const wrap_cond_v = wrap_cond<B,T>::value;
    
} // namespace tlib::simd::detail


#endif // TLIB_DETAIL_KERNEL_SIMD_SUPPORT_H
