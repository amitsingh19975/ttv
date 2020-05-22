#ifndef TLIB_DETAIL_KERNEL_SIMD_H
#define TLIB_DETAIL_KERNEL_SIMD_H

#include "simd_support.h"

namespace tlib::simd::detail{
    
    union VFloat{
        __m256 y;
        __m128 x[2];
    };

    union VDouble{
        __m256d y;
        __m128d x[2];
    };

    inline void fmadd(VFloat const& p1, VFloat const& p2, VFloat& acc){
        if constexpr( simd_config::fma ){
            acc.y = _mm256_fmadd_ps(p1.y,p2.y,acc.y);
        }else{
            auto temp = _mm256_mul_ps(p1.y,p2.y);
            acc.y = _mm256_add_ps(temp,acc.y);
        }
    }

    inline void fmadd(VFloat const& p1, VFloat const& p2, VFloat& acc, size_t idx){
        if constexpr( simd_config::fma ){
            acc.x[idx] = _mm_fmadd_ps(p1.x[idx],p2.x[idx],acc.x[idx]);
        }else{
            auto temp = _mm_mul_ps(p1.x[idx],p2.x[idx]);
            acc.x[idx] = _mm_add_ps(temp,acc.x[idx]);
        }
    }

    inline void fmadd(VDouble const& p1, VDouble const& p2, VDouble& acc){
        if constexpr( simd_config::fma ){
            acc.y = _mm256_fmadd_pd(p1.y,p2.y,acc.y);
        }else{
            auto temp = _mm256_mul_pd(p1.y,p2.y);
            acc.y = _mm256_add_pd(temp,acc.y);
        }
    }

    inline void fmadd(VDouble const& p1, VDouble const& p2, VDouble& acc, size_t idx){
        if constexpr( simd_config::fma ){
            acc.x[idx] = _mm_fmadd_pd(p1.x[idx],p2.x[idx],acc.x[idx]);
        }else{
            auto temp = _mm_mul_pd(p1.x[idx],p2.x[idx]);
            acc.x[idx] = _mm_add_pd(temp,acc.x[idx]);
        }
    }

} // namespace tlib::simd::detail



#endif // TLIB_DETAIL_KERNEL_SIMD_H
