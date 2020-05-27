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

    template<typename SizeType>
    inline void tran_16x16(__m512* res, float const* a, SizeType const& lda) noexcept{
        __m512i t[16];
        __m512i r[16];

        auto* mat = reinterpret_cast<uint32_t const*>(a);

        int mask;
        alignas(64) int64_t idx1[8] = {2, 3, 0, 1, 6, 7, 4, 5}; 
        alignas(64) int64_t idx2[8] = {1, 0, 3, 2, 5, 4, 7, 6}; 
        alignas(64) int32_t idx3[16] = {1, 0, 3, 2, 5 ,4 ,7 ,6 ,9 ,8 , 11, 10, 13, 12 ,15, 14};
        __m512i vidx1 = _mm512_load_epi64(idx1);
        __m512i vidx2 = _mm512_load_epi64(idx2);
        __m512i vidx3 = _mm512_load_epi32(idx3);

        t[0] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 0 * lda + 0] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 8 * lda + 0] ) ) , 1);
        t[1] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 1 * lda + 0] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 9 * lda + 0] ) ) , 1);
        t[2] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 2 * lda + 0] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[10 * lda + 0] ) ) , 1);
        t[3] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 3 * lda + 0] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[11 * lda + 0] ) ) , 1);
        t[4] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 4 * lda + 0] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[12 * lda + 0] ) ) , 1);
        t[5] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 5 * lda + 0] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[13 * lda + 0] ) ) , 1);
        t[6] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 6 * lda + 0] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[14 * lda + 0] ) ) , 1);
        t[7] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 7 * lda + 0] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[15 * lda + 0] ) ) , 1);

        t[8] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 0 * lda + 8] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 8 * lda + 8] ) ) , 1);
        t[9] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 1 * lda + 8] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 9 * lda + 8] ) ) , 1);
        t[10] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 2 * lda + 8] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[10 * lda + 8] ) ) , 1);
        t[11] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 3 * lda + 8] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[11 * lda + 8] ) ) , 1);
        t[12] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 4 * lda + 8] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[12 * lda + 8] ) ) , 1);
        t[13] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 5 * lda + 8] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[13 * lda + 8] ) ) , 1);
        t[14] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 6 * lda + 8] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[14 * lda + 8] ) ) , 1);
        t[15] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 7 * lda + 8] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[15 * lda + 8] ) ) , 1);

        mask= 0xcc;
        r[0] = _mm512_mask_permutexvar_epi64(t[0], static_cast<__mmask8>( mask ), vidx1, t[4]);
        r[1] = _mm512_mask_permutexvar_epi64(t[1], static_cast<__mmask8>( mask ), vidx1, t[5]);
        r[2] = _mm512_mask_permutexvar_epi64(t[2], static_cast<__mmask8>( mask ), vidx1, t[6]);
        r[3] = _mm512_mask_permutexvar_epi64(t[3], static_cast<__mmask8>( mask ), vidx1, t[7]);
        r[8] = _mm512_mask_permutexvar_epi64(t[8], static_cast<__mmask8>( mask ), vidx1, t[12]);
        r[9] = _mm512_mask_permutexvar_epi64(t[9], static_cast<__mmask8>( mask ), vidx1, t[13]);
        r[10] = _mm512_mask_permutexvar_epi64(t[10], static_cast<__mmask8>( mask ), vidx1, t[14]);
        r[11] = _mm512_mask_permutexvar_epi64(t[11], static_cast<__mmask8>( mask ), vidx1, t[15]);

        mask= 0x33;
        r[4] = _mm512_mask_permutexvar_epi64(t[4], static_cast<__mmask8>( mask ), vidx1, t[0]);
        r[5] = _mm512_mask_permutexvar_epi64(t[5], static_cast<__mmask8>( mask ), vidx1, t[1]);
        r[6] = _mm512_mask_permutexvar_epi64(t[6], static_cast<__mmask8>( mask ), vidx1, t[2]);
        r[7] = _mm512_mask_permutexvar_epi64(t[7], static_cast<__mmask8>( mask ), vidx1, t[3]);
        r[12] = _mm512_mask_permutexvar_epi64(t[12], static_cast<__mmask8>( mask ), vidx1, t[8]);
        r[13] = _mm512_mask_permutexvar_epi64(t[13], static_cast<__mmask8>( mask ), vidx1, t[9]);
        r[14] = _mm512_mask_permutexvar_epi64(t[14], static_cast<__mmask8>( mask ), vidx1, t[10]);
        r[15] = _mm512_mask_permutexvar_epi64(t[15], static_cast<__mmask8>( mask ), vidx1, t[11]);

        mask = 0xaa;
        t[0] = _mm512_mask_permutexvar_epi64(r[0], static_cast<__mmask8>( mask ), vidx2, r[2]);
        t[1] = _mm512_mask_permutexvar_epi64(r[1], static_cast<__mmask8>( mask ), vidx2, r[3]);
        t[4] = _mm512_mask_permutexvar_epi64(r[4], static_cast<__mmask8>( mask ), vidx2, r[6]);
        t[5] = _mm512_mask_permutexvar_epi64(r[5], static_cast<__mmask8>( mask ), vidx2, r[7]);
        t[8] = _mm512_mask_permutexvar_epi64(r[8], static_cast<__mmask8>( mask ), vidx2, r[10]);
        t[9] = _mm512_mask_permutexvar_epi64(r[9], static_cast<__mmask8>( mask ), vidx2, r[11]);
        t[12] = _mm512_mask_permutexvar_epi64(r[12], static_cast<__mmask8>( mask ), vidx2, r[14]);
        t[13] = _mm512_mask_permutexvar_epi64(r[13], static_cast<__mmask8>( mask ), vidx2, r[15]);

        mask = 0x55;
        t[2] = _mm512_mask_permutexvar_epi64(r[2], static_cast<__mmask8>( mask ), vidx2, r[0]);
        t[3] = _mm512_mask_permutexvar_epi64(r[3], static_cast<__mmask8>( mask ), vidx2, r[1]);
        t[6] = _mm512_mask_permutexvar_epi64(r[6], static_cast<__mmask8>( mask ), vidx2, r[4]);
        t[7] = _mm512_mask_permutexvar_epi64(r[7], static_cast<__mmask8>( mask ), vidx2, r[5]);
        t[10] = _mm512_mask_permutexvar_epi64(r[10], static_cast<__mmask8>( mask ), vidx2, r[8]);
        t[11] = _mm512_mask_permutexvar_epi64(r[11], static_cast<__mmask8>( mask ), vidx2, r[9]);
        t[14] = _mm512_mask_permutexvar_epi64(r[14], static_cast<__mmask8>( mask ), vidx2, r[12]);
        t[15] = _mm512_mask_permutexvar_epi64(r[15], static_cast<__mmask8>( mask ), vidx2, r[13]);

        mask = 0xaaaa;
        r[0] = _mm512_mask_permutexvar_epi32(t[0], static_cast<__mmask16>( mask ), vidx3, t[1]);
        r[2] = _mm512_mask_permutexvar_epi32(t[2], static_cast<__mmask16>( mask ), vidx3, t[3]);
        r[4] = _mm512_mask_permutexvar_epi32(t[4], static_cast<__mmask16>( mask ), vidx3, t[5]);
        r[6] = _mm512_mask_permutexvar_epi32(t[6], static_cast<__mmask16>( mask ), vidx3, t[7]);
        r[8] = _mm512_mask_permutexvar_epi32(t[8], static_cast<__mmask16>( mask ), vidx3, t[9]);
        r[10] = _mm512_mask_permutexvar_epi32(t[10], static_cast<__mmask16>( mask ), vidx3, t[11]);
        r[12] = _mm512_mask_permutexvar_epi32(t[12], static_cast<__mmask16>( mask ), vidx3, t[13]);
        r[14] = _mm512_mask_permutexvar_epi32(t[14], static_cast<__mmask16>( mask ), vidx3, t[15]);    

        mask = 0x5555;
        r[1] = _mm512_mask_permutexvar_epi32(t[1], static_cast<__mmask16>( mask ), vidx3, t[0]);
        r[3] = _mm512_mask_permutexvar_epi32(t[3], static_cast<__mmask16>( mask ), vidx3, t[2]);
        r[5] = _mm512_mask_permutexvar_epi32(t[5], static_cast<__mmask16>( mask ), vidx3, t[4]);
        r[7] = _mm512_mask_permutexvar_epi32(t[7], static_cast<__mmask16>( mask ), vidx3, t[6]);
        r[9] = _mm512_mask_permutexvar_epi32(t[9], static_cast<__mmask16>( mask ), vidx3, t[8]);  
        r[11] = _mm512_mask_permutexvar_epi32(t[11], static_cast<__mmask16>( mask ), vidx3, t[10]);  
        r[13] = _mm512_mask_permutexvar_epi32(t[13], static_cast<__mmask16>( mask ), vidx3, t[12]);
        r[15] = _mm512_mask_permutexvar_epi32(t[15], static_cast<__mmask16>( mask ), vidx3, t[14]);

        res[0] = _mm512_castsi512_ps(r[0]);
        res[1] = _mm512_castsi512_ps(r[1]);
        res[2] = _mm512_castsi512_ps(r[2]);
        res[3] = _mm512_castsi512_ps(r[3]);
        res[4] = _mm512_castsi512_ps(r[4]);
        res[5] = _mm512_castsi512_ps(r[5]);
        res[6] = _mm512_castsi512_ps(r[6]);
        res[7] = _mm512_castsi512_ps(r[7]);
        res[8] = _mm512_castsi512_ps(r[8]);
        res[9] = _mm512_castsi512_ps(r[9]);
        res[10] = _mm512_castsi512_ps(r[10]);
        res[11] = _mm512_castsi512_ps(r[11]);
        res[12] = _mm512_castsi512_ps(r[12]);
        res[13] = _mm512_castsi512_ps(r[13]);
        res[14] = _mm512_castsi512_ps(r[14]);
        res[15] = _mm512_castsi512_ps(r[15]);

    }

    template<typename SizeType>
    inline void tran_16x8(__m512* res, float const* a, SizeType const& lda) noexcept{
        __m512i t[8];
        __m512i r[8];

        auto* mat = reinterpret_cast<uint32_t const*>(a);

        int mask;
        alignas(64) int64_t idx1[8] = {2, 3, 0, 1, 6, 7, 4, 5}; 
        alignas(64) int64_t idx2[8] = {1, 0, 3, 2, 5, 4, 7, 6}; 
        alignas(64) int32_t idx3[16] = {1, 0, 3, 2, 5 ,4 ,7 ,6 ,9 ,8 , 11, 10, 13, 12 ,15, 14};
        __m512i vidx1 = _mm512_load_epi64(idx1);
        __m512i vidx2 = _mm512_load_epi64(idx2);
        __m512i vidx3 = _mm512_load_epi32(idx3);

        t[0] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 0 * lda + 0] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 8 * lda + 0] ) ) , 1);
        t[1] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 1 * lda + 0] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 9 * lda + 0] ) ) , 1);
        t[2] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 2 * lda + 0] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[10 * lda + 0] ) ) , 1);
        t[3] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 3 * lda + 0] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[11 * lda + 0] ) ) , 1);
        t[4] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 4 * lda + 0] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[12 * lda + 0] ) ) , 1);
        t[5] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 5 * lda + 0] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[13 * lda + 0] ) ) , 1);
        t[6] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 6 * lda + 0] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[14 * lda + 0] ) ) , 1);
        t[7] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 7 * lda + 0] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[15 * lda + 0] ) ) , 1);

        mask= 0xcc;
        r[0] = _mm512_mask_permutexvar_epi64(t[0], static_cast<__mmask8>( mask ), vidx1, t[4]);
        r[1] = _mm512_mask_permutexvar_epi64(t[1], static_cast<__mmask8>( mask ), vidx1, t[5]);
        r[2] = _mm512_mask_permutexvar_epi64(t[2], static_cast<__mmask8>( mask ), vidx1, t[6]);
        r[3] = _mm512_mask_permutexvar_epi64(t[3], static_cast<__mmask8>( mask ), vidx1, t[7]);

        mask= 0x33;
        r[4] = _mm512_mask_permutexvar_epi64(t[4], static_cast<__mmask8>( mask ), vidx1, t[0]);
        r[5] = _mm512_mask_permutexvar_epi64(t[5], static_cast<__mmask8>( mask ), vidx1, t[1]);
        r[6] = _mm512_mask_permutexvar_epi64(t[6], static_cast<__mmask8>( mask ), vidx1, t[2]);
        r[7] = _mm512_mask_permutexvar_epi64(t[7], static_cast<__mmask8>( mask ), vidx1, t[3]);

        mask = 0xaa;
        t[0] = _mm512_mask_permutexvar_epi64(r[0], static_cast<__mmask8>( mask ), vidx2, r[2]);
        t[1] = _mm512_mask_permutexvar_epi64(r[1], static_cast<__mmask8>( mask ), vidx2, r[3]);
        t[4] = _mm512_mask_permutexvar_epi64(r[4], static_cast<__mmask8>( mask ), vidx2, r[6]);
        t[5] = _mm512_mask_permutexvar_epi64(r[5], static_cast<__mmask8>( mask ), vidx2, r[7]);

        mask = 0x55;
        t[2] = _mm512_mask_permutexvar_epi64(r[2], static_cast<__mmask8>( mask ), vidx2, r[0]);
        t[3] = _mm512_mask_permutexvar_epi64(r[3], static_cast<__mmask8>( mask ), vidx2, r[1]);
        t[6] = _mm512_mask_permutexvar_epi64(r[6], static_cast<__mmask8>( mask ), vidx2, r[4]);
        t[7] = _mm512_mask_permutexvar_epi64(r[7], static_cast<__mmask8>( mask ), vidx2, r[5]);

        mask = 0xaaaa;
        r[0] = _mm512_mask_permutexvar_epi32(t[0], static_cast<__mmask16>( mask ), vidx3, t[1]);
        r[2] = _mm512_mask_permutexvar_epi32(t[2], static_cast<__mmask16>( mask ), vidx3, t[3]);
        r[4] = _mm512_mask_permutexvar_epi32(t[4], static_cast<__mmask16>( mask ), vidx3, t[5]);
        r[6] = _mm512_mask_permutexvar_epi32(t[6], static_cast<__mmask16>( mask ), vidx3, t[7]);    

        mask = 0x5555;
        r[1] = _mm512_mask_permutexvar_epi32(t[1], static_cast<__mmask16>( mask ), vidx3, t[0]);
        r[3] = _mm512_mask_permutexvar_epi32(t[3], static_cast<__mmask16>( mask ), vidx3, t[2]);
        r[5] = _mm512_mask_permutexvar_epi32(t[5], static_cast<__mmask16>( mask ), vidx3, t[4]);
        r[7] = _mm512_mask_permutexvar_epi32(t[7], static_cast<__mmask16>( mask ), vidx3, t[6]);

        res[0] = _mm512_castsi512_ps(r[0]);
        res[1] = _mm512_castsi512_ps(r[1]);
        res[2] = _mm512_castsi512_ps(r[2]);
        res[3] = _mm512_castsi512_ps(r[3]);
        res[4] = _mm512_castsi512_ps(r[4]);
        res[5] = _mm512_castsi512_ps(r[5]);
        res[6] = _mm512_castsi512_ps(r[6]);
        res[7] = _mm512_castsi512_ps(r[7]);

    }

    template<typename SizeType>
    inline void tran_16x4(__m512* res, float const* a, SizeType const& lda) noexcept{
        __m512i t[4];
        __m512i r[4];

        auto* mat = reinterpret_cast<uint32_t const*>(a);

        int mask;
        alignas(64) int64_t idx1[8] = {2, 3, 0, 1, 6, 7, 4, 5}; 
        alignas(64) int64_t idx2[8] = {1, 0, 3, 2, 5, 4, 7, 6}; 
        alignas(64) int32_t idx3[16] = {1, 0, 3, 2, 5 ,4 ,7 ,6 ,9 ,8 , 11, 10, 13, 12 ,15, 14};
        __m512i vidx1 = _mm512_load_epi64(idx1);
        __m512i vidx2 = _mm512_load_epi64(idx2);
        __m512i vidx3 = _mm512_load_epi32(idx3);

        t[0] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 0 * lda + 0] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 2 * lda + 0] ) ) , 1);
        t[1] = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 1 * lda + 0] ) ) ), _mm256_loadu_si256( reinterpret_cast<__m256i const*> ( &mat[ 3 * lda + 0] ) ) , 1);

        mask= 0xcc;
        r[0] = _mm512_mask_permutexvar_epi64(t[0], static_cast<__mmask8>( mask ), vidx1, t[2]);
        r[1] = _mm512_mask_permutexvar_epi64(t[1], static_cast<__mmask8>( mask ), vidx1, t[3]);

        mask= 0x33;
        r[2] = _mm512_mask_permutexvar_epi64(t[2], static_cast<__mmask8>( mask ), vidx1, t[0]);
        r[3] = _mm512_mask_permutexvar_epi64(t[3], static_cast<__mmask8>( mask ), vidx1, t[1]);

        mask = 0xaa;
        t[0] = _mm512_mask_permutexvar_epi64(r[0], static_cast<__mmask8>( mask ), vidx2, r[2]);
        t[1] = _mm512_mask_permutexvar_epi64(r[1], static_cast<__mmask8>( mask ), vidx2, r[3]);

        mask = 0x55;
        t[2] = _mm512_mask_permutexvar_epi64(r[2], static_cast<__mmask8>( mask ), vidx2, r[0]);
        t[3] = _mm512_mask_permutexvar_epi64(r[3], static_cast<__mmask8>( mask ), vidx2, r[1]);

        mask = 0xaaaa;
        r[0] = _mm512_mask_permutexvar_epi32(t[0], static_cast<__mmask16>( mask ), vidx3, t[1]);
        r[2] = _mm512_mask_permutexvar_epi32(t[2], static_cast<__mmask16>( mask ), vidx3, t[3]);   

        mask = 0x5555;
        r[1] = _mm512_mask_permutexvar_epi32(t[1], static_cast<__mmask16>( mask ), vidx3, t[0]);
        r[3] = _mm512_mask_permutexvar_epi32(t[3], static_cast<__mmask16>( mask ), vidx3, t[2]);

        res[0] = _mm512_castsi512_ps(r[0]);
        res[1] = _mm512_castsi512_ps(r[1]);
        res[2] = _mm512_castsi512_ps(r[2]);
        res[3] = _mm512_castsi512_ps(r[3]);

    }

    inline void tran_4x4_helper(__m256d& a, __m256d& b, __m256d& c, __m256d& d ) noexcept{
        __m128d temp[8] ;
        temp[0] = _mm_unpacklo_pd( _mm256_extractf128_pd( a, 0 ), _mm256_extractf128_pd( b, 0 ) );
        temp[1] = _mm_unpacklo_pd( _mm256_extractf128_pd( c, 0 ), _mm256_extractf128_pd( d, 0 ) );
        temp[2] = _mm_unpackhi_pd( _mm256_extractf128_pd( a, 0 ), _mm256_extractf128_pd( b, 0 ) );
        temp[3] = _mm_unpackhi_pd( _mm256_extractf128_pd( c, 0 ), _mm256_extractf128_pd( d, 0 ) );
        
        temp[4] = _mm_unpacklo_pd( _mm256_extractf128_pd( a, 1 ), _mm256_extractf128_pd( b, 1 ) );
        temp[5] = _mm_unpacklo_pd( _mm256_extractf128_pd( c, 1 ), _mm256_extractf128_pd( d, 1 ) );
        temp[6] = _mm_unpackhi_pd( _mm256_extractf128_pd( a, 1 ), _mm256_extractf128_pd( b, 1 ) );
        temp[7] = _mm_unpackhi_pd( _mm256_extractf128_pd( c, 1 ), _mm256_extractf128_pd( d, 1 ) );
        
        a = _mm256_insertf128_pd( _mm256_castpd128_pd256(_mm_blend_pd(temp[0],temp[1],_MM_SHUFFLE2(0,0))), _mm_blend_pd(temp[0],temp[1],_MM_SHUFFLE2(1,1)), 1 );
        b = _mm256_insertf128_pd( _mm256_castpd128_pd256(_mm_blend_pd(temp[2],temp[3],_MM_SHUFFLE2(0,0))), _mm_blend_pd(temp[2],temp[3],_MM_SHUFFLE2(1,1)), 1 );
        c = _mm256_insertf128_pd( _mm256_castpd128_pd256(_mm_blend_pd(temp[4],temp[5],_MM_SHUFFLE2(0,0))), _mm_blend_pd(temp[4],temp[5],_MM_SHUFFLE2(1,1)), 1 );
        d = _mm256_insertf128_pd( _mm256_castpd128_pd256(_mm_blend_pd(temp[6],temp[7],_MM_SHUFFLE2(0,0))), _mm_blend_pd(temp[6],temp[7],_MM_SHUFFLE2(1,1)), 1 );

    }

    template<typename SizeType>
    inline void tran_8x8(__m512d* res, double const* a, SizeType const& lda) noexcept{

        __m256d temp[8];

        temp[0] = _mm256_loadu_pd(a + lda * 0 + 0);
        temp[1] = _mm256_loadu_pd(a + lda * 1 + 0);
        temp[2] = _mm256_loadu_pd(a + lda * 2 + 0);
        temp[3] = _mm256_loadu_pd(a + lda * 3 + 0);
        
        temp[4] = _mm256_loadu_pd(a + lda * 4 + 0);
        temp[5] = _mm256_loadu_pd(a + lda * 5 + 0);
        temp[6] = _mm256_loadu_pd(a + lda * 6 + 0);
        temp[7] = _mm256_loadu_pd(a + lda * 7 + 0);

        tran_4x4_helper(temp[0], temp[1], temp[2], temp[3]);
        tran_4x4_helper(temp[4], temp[5], temp[6], temp[7]);
        
        res[0] = _mm512_insertf64x4( _mm512_castpd256_pd512(temp[0]), temp[4], 1 );
        res[1] = _mm512_insertf64x4( _mm512_castpd256_pd512(temp[1]), temp[5], 1 );
        res[2] = _mm512_insertf64x4( _mm512_castpd256_pd512(temp[2]), temp[6], 1 );
        res[3] = _mm512_insertf64x4( _mm512_castpd256_pd512(temp[3]), temp[7], 1 );

        temp[0] = _mm256_loadu_pd(a + lda * 0 + 4);
        temp[1] = _mm256_loadu_pd(a + lda * 1 + 4);
        temp[2] = _mm256_loadu_pd(a + lda * 2 + 4);
        temp[3] = _mm256_loadu_pd(a + lda * 3 + 4);
        
        temp[4] = _mm256_loadu_pd(a + lda * 4 + 4);
        temp[5] = _mm256_loadu_pd(a + lda * 5 + 4);
        temp[6] = _mm256_loadu_pd(a + lda * 6 + 4);
        temp[7] = _mm256_loadu_pd(a + lda * 7 + 4);

        tran_4x4_helper(temp[0], temp[1], temp[2], temp[3]);
        tran_4x4_helper(temp[4], temp[5], temp[6], temp[7]);

                
        res[4] = _mm512_insertf64x4( _mm512_castpd256_pd512(temp[0]), temp[4], 1 );
        res[5] = _mm512_insertf64x4( _mm512_castpd256_pd512(temp[1]), temp[5], 1 );
        res[6] = _mm512_insertf64x4( _mm512_castpd256_pd512(temp[2]), temp[6], 1 );
        res[7] = _mm512_insertf64x4( _mm512_castpd256_pd512(temp[3]), temp[7], 1 );

    }

    template<typename SizeType>
    inline void tran_4x4(__m512d* res, double const* a, SizeType const& lda) noexcept{

        __m256d temp[8];

        temp[0] = _mm256_loadu_pd(a + lda * 0 + 0);
        temp[1] = _mm256_loadu_pd(a + lda * 1 + 0);
        temp[2] = _mm256_loadu_pd(a + lda * 2 + 0);
        temp[3] = _mm256_loadu_pd(a + lda * 3 + 0);
        
        temp[4] = _mm256_loadu_pd(a + lda * 4 + 0);
        temp[5] = _mm256_loadu_pd(a + lda * 5 + 0);
        temp[6] = _mm256_loadu_pd(a + lda * 6 + 0);
        temp[7] = _mm256_loadu_pd(a + lda * 7 + 0);

        tran_4x4_helper(temp[0], temp[1], temp[2], temp[3]);
        tran_4x4_helper(temp[4], temp[5], temp[6], temp[7]);
        
        res[0] = _mm512_insertf64x4( _mm512_castpd256_pd512(temp[0]), temp[4], 1 );
        res[1] = _mm512_insertf64x4( _mm512_castpd256_pd512(temp[1]), temp[5], 1 );
        res[2] = _mm512_insertf64x4( _mm512_castpd256_pd512(temp[2]), temp[6], 1 );
        res[3] = _mm512_insertf64x4( _mm512_castpd256_pd512(temp[3]), temp[7], 1 );

    }

} // namespace tlib::simd::detail



#endif // TLIB_DETAIL_KERNEL_SIMD_H
