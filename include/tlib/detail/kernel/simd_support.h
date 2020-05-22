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

        #if defined(__FMA__)
            static constexpr bool fma = true;
        #else
            static constexpr bool fma = false;
        #endif

    };
    
} // namespace tlib::simd::detail


#endif // TLIB_DETAIL_KERNEL_SIMD_SUPPORT_H
