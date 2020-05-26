#ifndef TLIB_DETAIL_KERNEL_X86_ROW_H
#define TLIB_DETAIL_KERNEL_X86_ROW_H

#include <cstddef>
#include "simd.h"
#include "layout.h"

namespace tlib::simd::x86{
    
    template<typename>
    struct kernel;

    namespace sse{
        template<typename,typename>
        struct kernel_helper;
    } // avx256

    namespace avx256{
        template<typename,typename>
        struct kernel_helper;
    } // avx256

    namespace avx512{
        template<typename,typename>
        struct kernel_helper;
    } // avx256


} // namespace tlib::simd


namespace tlib::simd::x86::avx256{
    template<>
    struct kernel_helper<float,row_major>{
        
        static constexpr std::size_t M = 8;
        static constexpr std::size_t K = 8;

        template<typename SizeType>
        inline void operator()(float* c, SizeType const* nc, SizeType const* wc,
            float const* a, SizeType const* na, SizeType const* wa,
            float const* b, SizeType const* nb, SizeType const* wb
        ) const noexcept{

            static_assert(
                detail::wrap_cond_v< detail::simd_config::avx , SizeType> && 
                detail::wrap_cond_v< detail::simd_config::avx2 , SizeType> && 
                "Your processor does not support AVX and AVX2 instruction set"
            );

            detail::VFloat r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, res;
            auto k_iter = na[1] / K;
            auto k_rem = na[1] % K;

            auto ak = a;
            auto bk = b;
            auto ck = c;
            
            res.y = _mm256_loadu_ps(ck);

            while(k_iter--){

                r0.x[0] = _mm_loadu_ps(ak + wa[0] * 0 + 0);
                r0.x[1] = _mm_loadu_ps(ak + wa[0] * 4 + 0);
                
                r1.x[0] = _mm_loadu_ps(ak + wa[0] * 1 + 0);
                r1.x[1] = _mm_loadu_ps(ak + wa[0] * 5 + 0);
                
                r2.x[0] = _mm_loadu_ps(ak + wa[0] * 2 + 0);
                r2.x[1] = _mm_loadu_ps(ak + wa[0] * 6 + 0);
                
                r3.x[0] = _mm_loadu_ps(ak + wa[0] * 3 + 0);
                r3.x[1] = _mm_loadu_ps(ak + wa[0] * 7 + 0);
                
                r4.x[0] = _mm_loadu_ps(ak + wa[0] * 0 + 4);
                r4.x[1] = _mm_loadu_ps(ak + wa[0] * 4 + 4);
                
                r5.x[0] = _mm_loadu_ps(ak + wa[0] * 1 + 4);
                r5.x[1] = _mm_loadu_ps(ak + wa[0] * 5 + 4);
                
                r6.x[0] = _mm_loadu_ps(ak + wa[0] * 2 + 4);
                r6.x[1] = _mm_loadu_ps(ak + wa[0] * 6 + 4);
                
                r7.x[0] = _mm_loadu_ps(ak + wa[0] * 3 + 4);
                r7.x[1] = _mm_loadu_ps(ak + wa[0] * 7 + 4);

                r8.y = _mm256_unpacklo_ps(r0.y,r1.y);
                r9.y = _mm256_unpackhi_ps(r0.y,r1.y);
                r10.y = _mm256_unpacklo_ps(r2.y,r3.y);
                r11.y = _mm256_unpackhi_ps(r2.y,r3.y);

                r12.y = _mm256_shuffle_ps(r8.y,r10.y,0x4E);
                r0.y = _mm256_blend_ps(r8.y,r12.y,0xCC);
                r1.y = _mm256_blend_ps(r10.y,r12.y,0x33);

                r12.y = _mm256_shuffle_ps(r9.y,r11.y,0x4E);
                r2.y = _mm256_blend_ps(r9.y,r12.y,0xCC);
                r3.y = _mm256_blend_ps(r11.y,r12.y,0x33);

                r8.y = _mm256_unpacklo_ps(r4.y,r5.y);
                r9.y = _mm256_unpackhi_ps(r4.y,r5.y);
                r10.y = _mm256_unpacklo_ps(r6.y,r7.y);
                r11.y = _mm256_unpackhi_ps(r6.y,r7.y);

                r12.y = _mm256_shuffle_ps(r8.y,r10.y,0x4E);
                r4.y = _mm256_blend_ps(r8.y,r12.y,0xCC);
                r5.y = _mm256_blend_ps(r10.y,r12.y,0x33);

                r12.y = _mm256_shuffle_ps(r9.y,r11.y,0x4E);
                r6.y = _mm256_blend_ps(r9.y,r12.y,0xCC);
                r7.y = _mm256_blend_ps(r11.y,r12.y,0x33);

                r9.y = _mm256_loadu_ps(bk);

                r10.y = _mm256_permute2f128_ps(r9.y,r9.y,0);
                r11.y = _mm256_permute2f128_ps(r9.y,r9.y,0x11);

                r8.y = _mm256_permute_ps(r10.y,_MM_SHUFFLE(0,0,0,0));
                detail::fmadd(r0,r8,res);

                r8.y = _mm256_permute_ps(r10.y,_MM_SHUFFLE(1,1,1,1));
                detail::fmadd(r1,r8,res);

                r8.y = _mm256_permute_ps(r10.y,_MM_SHUFFLE(2,2,2,2));
                detail::fmadd(r2,r8,res);

                r8.y = _mm256_permute_ps(r10.y,_MM_SHUFFLE(3,3,3,3));
                detail::fmadd(r3,r8,res);

                r8.y = _mm256_permute_ps(r11.y,_MM_SHUFFLE(0,0,0,0));
                detail::fmadd(r4,r8,res);

                r8.y = _mm256_permute_ps(r11.y,_MM_SHUFFLE(1,1,1,1));
                detail::fmadd(r5,r8,res);

                r8.y = _mm256_permute_ps(r11.y,_MM_SHUFFLE(2,2,2,2));
                detail::fmadd(r6,r8,res);

                r8.y = _mm256_permute_ps(r11.y,_MM_SHUFFLE(3,3,3,3));
                detail::fmadd(r7,r8,res);

                ak += wa[1] * K;
                bk += K;
            }

            k_iter = k_rem / 4;
            k_rem = k_rem % 4;

            if( k_iter ){

                r0.x[0] = _mm_loadu_ps(ak + wa[0] * 0 + 0);
                
                r1.x[0] = _mm_loadu_ps(ak + wa[0] * 1 + 0);
                
                r2.x[0] = _mm_loadu_ps(ak + wa[0] * 2 + 0);
                
                r3.x[0] = _mm_loadu_ps(ak + wa[0] * 3 + 0);
                
                r0.x[1] = _mm_loadu_ps(ak + wa[0] * 4 + 0);
                
                r1.x[1] = _mm_loadu_ps(ak + wa[0] * 5 + 0);
                
                r2.x[1] = _mm_loadu_ps(ak + wa[0] * 6 + 0);
                
                r3.x[1] = _mm_loadu_ps(ak + wa[0] * 7 + 0);

                _MM_TRANSPOSE4_PS(r0.x[0], r1.x[0],r2.x[0], r3.x[0]);
                _MM_TRANSPOSE4_PS(r0.x[1], r1.x[1],r2.x[1], r3.x[1]);

                r9.y = _mm256_loadu_ps(bk);

                r10.y = _mm256_permute2f128_ps(r9.y,r9.y,0);
                r11.y = _mm256_permute2f128_ps(r9.y,r9.y,0x11);

                 r8.y = _mm256_permute_ps(r10.y,_MM_SHUFFLE(0,0,0,0));
                detail::fmadd(r0,r8,res);

                r8.y = _mm256_permute_ps(r10.y,_MM_SHUFFLE(1,1,1,1));
                detail::fmadd(r1,r8,res);

                r8.y = _mm256_permute_ps(r10.y,_MM_SHUFFLE(2,2,2,2));
                detail::fmadd(r2,r8,res);

                r8.y = _mm256_permute_ps(r10.y,_MM_SHUFFLE(3,3,3,3));
                detail::fmadd(r3,r8,res);

                ak += wa[1] * 4;
                bk += 4;
            }
            
            _mm256_storeu_ps(ck, res.y);

            while(k_rem--){
                auto ai = ak;
                auto bi = bk;
                auto ci = ck;
                #pragma omp simd
                for( auto i = 0; i < M; ++i ){
                    *ci += *ai * *bi;
                    ++ci;
                    ai += wa[0];
                }
                ak += wa[1];
                ++bk;
            }

        }

    };
    
    template<>
    struct kernel_helper<double,row_major>{
        
        static constexpr std::size_t M = 4;
        static constexpr std::size_t K = 4;

        template<typename SizeType>
        inline void operator()(double* c, SizeType const* nc, SizeType const* wc,
            double const* a, SizeType const* na, SizeType const* wa,
            double const* b, SizeType const* nb, SizeType const* wb
        ) const noexcept{

            static_assert(
                detail::wrap_cond_v< detail::simd_config::avx , SizeType> && 
                detail::wrap_cond_v< detail::simd_config::avx2 , SizeType> && 
                "Your processor does not support AVX and AVX2 instruction set"
            );

            detail::VDouble r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, res;
            auto k_iter = na[1] / K;
            auto k_rem = na[1] % K;

            auto ak = a;
            auto bk = b;
            auto ck = c;
            
            res.y = _mm256_loadu_pd(ck);

            while(k_iter--){
                r0.y = _mm256_loadu_pd(ak + wa[0] * 0);
                r1.y = _mm256_loadu_pd(ak + wa[0] * 1);
                r2.y = _mm256_loadu_pd(ak + wa[0] * 2);
                r3.y = _mm256_loadu_pd(ak + wa[0] * 3);

                r4.x[0] = _mm_unpacklo_pd(r0.x[0], r1.x[0]);
                r5.x[0] = _mm_unpacklo_pd(r2.x[0], r3.x[0]);
                r6.x[0] = _mm_unpackhi_pd(r0.x[0], r1.x[0]);
                r7.x[0] = _mm_unpackhi_pd(r2.x[0], r3.x[0]);
                
                r8.x[0] = _mm_unpacklo_pd(r0.x[1], r1.x[1]);
                r9.x[0] = _mm_unpacklo_pd(r2.x[1], r3.x[1]);
                r10.x[0] = _mm_unpackhi_pd(r0.x[1], r1.x[1]);
                r11.x[0] = _mm_unpackhi_pd(r2.x[1], r3.x[1]);

                r0.x[0] = _mm_blend_pd(r4.x[0],r5.x[0],_MM_SHUFFLE2(0,0));
                r0.x[1] = _mm_blend_pd(r4.x[0],r5.x[0],_MM_SHUFFLE2(1,1));
                
                r1.x[0] = _mm_blend_pd(r6.x[0],r7.x[0],_MM_SHUFFLE2(0,0));
                r1.x[1] = _mm_blend_pd(r6.x[0],r7.x[0],_MM_SHUFFLE2(1,1));
                
                r2.x[0] = _mm_blend_pd(r8.x[0],r9.x[0],_MM_SHUFFLE2(0,0));
                r2.x[1] = _mm_blend_pd(r8.x[0],r9.x[0],_MM_SHUFFLE2(1,1));
                
                r3.x[0] = _mm_blend_pd(r10.x[0],r11.x[0],_MM_SHUFFLE2(0,0));
                r3.x[1] = _mm_blend_pd(r10.x[0],r11.x[0],_MM_SHUFFLE2(1,1));

                r8.y = _mm256_broadcast_sd(bk);
                detail::fmadd(r0,r8,res);

                r8.y = _mm256_broadcast_sd(bk + 1);
                detail::fmadd(r1,r8,res);

                r8.y = _mm256_broadcast_sd(bk + 2);
                detail::fmadd(r2,r8,res);

                r8.y = _mm256_broadcast_sd(bk + 3);
                detail::fmadd(r3,r8,res);

                ak += wa[1] * K;
                bk += K;
            }
            
            _mm256_storeu_pd(ck, res.y);

            while(k_rem--){
                auto ai = ak;
                auto bi = bk;
                auto ci = ck;
                #pragma omp simd
                for( auto i = 0; i < M; ++i ){
                    *ci += *ai * *bi;
                    ++ci;
                    ai += wa[0];
                }
                ak += wa[1];
                ++bk;
            }

        }

    };
    
    
} // namespace tlib::simd::x86::avx256

namespace tlib::simd::x86::sse{
    template<>
    struct kernel_helper<float,row_major>{
        
        static constexpr std::size_t M = 4;
        static constexpr std::size_t K = 4;

        template<typename SizeType>
        inline void operator()(float* c, SizeType const* nc, SizeType const* wc,
            float const* a, SizeType const* na, SizeType const* wa,
            float const* b, SizeType const* nb, SizeType const* wb
        ) const noexcept{

            static_assert(
                detail::wrap_cond_v< detail::simd_config::sse , SizeType> && 
                detail::wrap_cond_v< detail::simd_config::sse4_1 , SizeType> && 
                detail::wrap_cond_v< detail::simd_config::sse2 , SizeType> && 
                detail::wrap_cond_v< detail::simd_config::sse3, SizeType >,
                "Your processor does not support SSE instruction set"
            );
            detail::VFloat r0, r1, r2, r3, r4, r8, r10, r11, res;

            auto k_iter = na[1] / K;
            auto k_rem = na[1] % K;

            auto ak = a;
            auto bk = b;
            auto ck = c;
            
            res.x[0] = _mm_loadu_ps(ck);

            while(k_iter--){
                r0.x[0] = _mm_loadu_ps(ak);
                r1.x[0] = _mm_loadu_ps(ak + wa[0]);
                r2.x[0] = _mm_loadu_ps(ak + wa[0] * 2);
                r3.x[0] = _mm_loadu_ps(ak + wa[0] * 3);

                _MM_TRANSPOSE4_PS(r0.x[0], r1.x[0], r2.x[0], r3.x[0]);

                r8.x[0] = _mm_loadu_ps(bk);

                r11.x[0] = _mm_permute_ps(r8.x[0], _MM_SHUFFLE(0,0,0,0));
                detail::fmadd(r0,r11,res,0);
                
                r11.x[0] = _mm_permute_ps(r8.x[0], _MM_SHUFFLE(1,1,1,1));
                detail::fmadd(r1,r11,res,0);
                
                r11.x[0] = _mm_permute_ps(r8.x[0], _MM_SHUFFLE(2,2,2,2));
                detail::fmadd(r2,r11,res,0);
                
                r11.x[0] = _mm_permute_ps(r8.x[0], _MM_SHUFFLE(3,3,3,3));
                detail::fmadd(r3,r11,res,0);

                ak += wa[1] * K;
                bk += K;
            }

            _mm_storeu_ps(ck, res.x[0]);
            
            while(k_rem--){
                auto ai = ak;
                auto bi = bk;
                auto ci = ck;
                #pragma omp simd
                for( auto i = 0; i < M; ++i ){
                    *ci += *ai * *bi;
                    ++ci;
                    ai += wa[0];
                }
                ak += wa[1];
                ++bk;
            }
        }

    };

    template<>
    struct kernel_helper<double,row_major>{
        
        static constexpr std::size_t M = 2;
        static constexpr std::size_t K = 2;

        template<typename SizeType>
        inline void operator()(double* c, SizeType const* nc, SizeType const* wc,
            double const* a, SizeType const* na, SizeType const* wa,
            double const* b, SizeType const* nb, SizeType const* wb
        ) const noexcept{

            static_assert(
                detail::wrap_cond_v< detail::simd_config::sse , SizeType> && 
                detail::wrap_cond_v< detail::simd_config::sse4_1 , SizeType> && 
                detail::wrap_cond_v< detail::simd_config::sse2 , SizeType> && 
                detail::wrap_cond_v< detail::simd_config::sse3, SizeType>,
                "Your processor does not support SSE instruction set"
            );
            detail::VDouble r0, r1, r4, r8, r10, r11, res;

            auto k_iter = na[1];

            auto ak = a;
            auto bk = b;
            auto ck = c;
            double sum = 0;

            #pragma omp simd reduction(+:sum)
            for(auto i = 0ul; i < M; ++i){
                auto aj = ak;
                auto bj = bk;
                auto cj = ck;
                for( auto j = 0ul; j < na[1]; ++j ){
                    sum += *aj * *bj;
                    ++aj;
                    ++bj;
                }
                *ck += sum;
                sum = 0;
                ak += wa[0];
                ++ck;
            }

            // while(k_iter--){
            //     auto ai = ak;
            //     auto bi = bk;
            //     auto ci = ck;
            //     for( auto i = 0; i < M; ++i ){
            //         *ci += *ai * *bi;
            //         ++ci;
            //         ai += wa[0];
            //     }
            //     ak += wa[1];
            //     ++bk;
            // }
        }

    };

} // namespace tlib::simd::x86::sse

namespace tlib::simd::x86::avx512{
    
    template<>
    struct kernel_helper<float,row_major>{
        static constexpr std::size_t M = 16;
        static constexpr std::size_t K = 16;

        template<typename SizeType>
        inline void operator()(float* c, SizeType const* nc, SizeType const* wc,
            float const* a, SizeType const* na, SizeType const* wa,
            float const* b, SizeType const* nb, SizeType const* wb
        ) const noexcept{
            static_assert(
                detail::wrap_cond_v< detail::simd_config::avx512f , SizeType> ,
                "Your processor does not support AVX512F instruction set"
            );

            __m512 zmm[32], res;

            auto k_iter = na[1] / K;
            auto k_rem = na[1] % K;

            auto ak = a;
            auto bk = b;
            auto ck = c;
            
            __mmask8 mask1 = 0;
            __mmask16 mask2 = 0;
            __m512i idx1 = _mm512_set_epi64(2, 3, 0, 1, 6, 7, 4, 5);
            __m512i idx2 = _mm512_set_epi64(1, 0, 3, 2, 5, 4, 7, 6);
            __m512i idx3 = _mm512_set_epi32(1, 0, 3, 2, 5 ,4 ,7 ,6 ,9 ,8 , 11, 10, 13, 12 ,15, 14);

            res = _mm512_loadu_ps(ck);

            while(k_iter--){

                zmm[0] = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_load_ps( ak + wa[0] * 0 + 0 )), _mm256_load_ps( ak + wa[0] * 8 + 0 ),1);
                zmm[1] = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_load_ps( ak + wa[0] * 1 + 0 )), _mm256_load_ps( ak + wa[0] * 9 + 0 ),1);
                zmm[2] = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_load_ps( ak + wa[0] * 2 + 0 )), _mm256_load_ps( ak + wa[0] * 10 + 0 ),1);
                zmm[3] = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_load_ps( ak + wa[0] * 3 + 0 )), _mm256_load_ps( ak + wa[0] * 11 + 0 ),1);
                zmm[4] = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_load_ps( ak + wa[0] * 4 + 0 )), _mm256_load_ps( ak + wa[0] * 12 + 0 ),1);
                zmm[5] = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_load_ps( ak + wa[0] * 5 + 0 )), _mm256_load_ps( ak + wa[0] * 13 + 0 ),1);
                zmm[6] = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_load_ps( ak + wa[0] * 6 + 0 )), _mm256_load_ps( ak + wa[0] * 14 + 0 ),1);
                zmm[7] = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_load_ps( ak + wa[0] * 7 + 0 )), _mm256_load_ps( ak + wa[0] * 15 + 0 ),1);

                zmm[8] = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_load_ps( ak + wa[0] * 0 + 8 )), _mm256_load_ps( ak + wa[0] * 8 + 8 ),1);
                zmm[9] = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_load_ps( ak + wa[0] * 1 + 8 )), _mm256_load_ps( ak + wa[0] * 9 + 8 ),1);
                zmm[10] = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_load_ps( ak + wa[0] * 2 + 8 )), _mm256_load_ps( ak + wa[0] * 10 + 8 ),1);
                zmm[11] = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_load_ps( ak + wa[0] * 3 + 8 )), _mm256_load_ps( ak + wa[0] * 11 + 8 ),1);
                zmm[12] = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_load_ps( ak + wa[0] * 4 + 8 )), _mm256_load_ps( ak + wa[0] * 12 + 8 ),1);
                zmm[13] = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_load_ps( ak + wa[0] * 5 + 8 )), _mm256_load_ps( ak + wa[0] * 13 + 8 ),1);
                zmm[14] = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_load_ps( ak + wa[0] * 6 + 8 )), _mm256_load_ps( ak + wa[0] * 14 + 8 ),1);
                zmm[15] = _mm512_insertf64x4(_mm512_castpd256_pd512(_mm256_load_ps( ak + wa[0] * 7 + 8 )), _mm256_load_ps( ak + wa[0] * 15 + 8 ),1);

                mask1 = 0xcc;

                zmm[16] = _mm512_mask_permutexvar_ps(zmm[0], mask1,idx1,zmm[4]);
                zmm[17] = _mm512_mask_permutexvar_ps(zmm[1], mask1,idx1,zmm[5]);
                zmm[18] = _mm512_mask_permutexvar_ps(zmm[2], mask1,idx1,zmm[6]);
                zmm[19] = _mm512_mask_permutexvar_ps(zmm[3], mask1,idx1,zmm[7]);
                zmm[20] = _mm512_mask_permutexvar_ps(zmm[8], mask1,idx1,zmm[12]);
                zmm[21] = _mm512_mask_permutexvar_ps(zmm[9], mask1,idx1,zmm[13]);
                zmm[22] = _mm512_mask_permutexvar_ps(zmm[10], mask1,idx1,zmm[14]);
                zmm[23] = _mm512_mask_permutexvar_ps(zmm[11], mask1,idx1,zmm[15]);

                mask1 = 0x33;

                zmm[24] = _mm512_mask_permutexvar_ps(zmm[4], mask1,idx1,zmm[0]);
                zmm[25] = _mm512_mask_permutexvar_ps(zmm[5], mask1,idx1,zmm[1]);
                zmm[26] = _mm512_mask_permutexvar_ps(zmm[6], mask1,idx1,zmm[2]);
                zmm[27] = _mm512_mask_permutexvar_ps(zmm[7], mask1,idx1,zmm[3]);
                zmm[28] = _mm512_mask_permutexvar_ps(zmm[12], mask1,idx1,zmm[8]);
                zmm[29] = _mm512_mask_permutexvar_ps(zmm[13], mask1,idx1,zmm[9]);
                zmm[30] = _mm512_mask_permutexvar_ps(zmm[14], mask1,idx1,zmm[10]);
                zmm[31] = _mm512_mask_permutexvar_ps(zmm[15], mask1,idx1,zmm[11]);

                mask1 = 0xaa;

                zmm[0] = _mm512_mask_permutexvar_ps(zmm[16], mask1,idx2,zmm[18]);
                zmm[1] = _mm512_mask_permutexvar_ps(zmm[17], mask1,idx2,zmm[19]);
                zmm[2] = _mm512_mask_permutexvar_ps(zmm[20], mask1,idx2,zmm[22]);
                zmm[3] = _mm512_mask_permutexvar_ps(zmm[21], mask1,idx2,zmm[23]);
                zmm[4] = _mm512_mask_permutexvar_ps(zmm[24], mask1,idx2,zmm[26]);
                zmm[5] = _mm512_mask_permutexvar_ps(zmm[25], mask1,idx2,zmm[27]);
                zmm[6] = _mm512_mask_permutexvar_ps(zmm[28], mask1,idx2,zmm[30]);
                zmm[7] = _mm512_mask_permutexvar_ps(zmm[19], mask1,idx2,zmm[31]);

                mask1 = 0x55;

                zmm[8] = _mm512_mask_permutexvar_ps(zmm[18], mask1,idx2,zmm[16]);
                zmm[9] = _mm512_mask_permutexvar_ps(zmm[19], mask1,idx2,zmm[17]);
                zmm[10] = _mm512_mask_permutexvar_ps(zmm[22], mask1,idx2,zmm[20]);
                zmm[11] = _mm512_mask_permutexvar_ps(zmm[23], mask1,idx2,zmm[21]);
                zmm[12] = _mm512_mask_permutexvar_ps(zmm[26], mask1,idx2,zmm[24]);
                zmm[13] = _mm512_mask_permutexvar_ps(zmm[27], mask1,idx2,zmm[25]);
                zmm[14] = _mm512_mask_permutexvar_ps(zmm[30], mask1,idx2,zmm[28]);
                zmm[15] = _mm512_mask_permutexvar_ps(zmm[31], mask1,idx2,zmm[19]);


                mask2 = 0xaaaa;

                zmm[16] = _mm512_mask_permutexvar_ps(zmm[0], mask2,idx3,zmm[1]);
                zmm[17] = _mm512_mask_permutexvar_ps(zmm[2], mask2,idx3,zmm[3]);
                zmm[18] = _mm512_mask_permutexvar_ps(zmm[4], mask2,idx3,zmm[5]);
                zmm[19] = _mm512_mask_permutexvar_ps(zmm[6], mask2,idx3,zmm[7]);
                zmm[20] = _mm512_mask_permutexvar_ps(zmm[8], mask2,idx3,zmm[9]);
                zmm[21] = _mm512_mask_permutexvar_ps(zmm[10], mask2,idx3,zmm[11]);
                zmm[22] = _mm512_mask_permutexvar_ps(zmm[12], mask2,idx3,zmm[13]);
                zmm[23] = _mm512_mask_permutexvar_ps(zmm[14], mask2,idx3,zmm[15]);

                mask2 = 0xaaaa;

                zmm[24] = _mm512_mask_permutexvar_ps(zmm[1], mask2,idx3,zmm[0]);
                zmm[25] = _mm512_mask_permutexvar_ps(zmm[3], mask2,idx3,zmm[2]);
                zmm[26] = _mm512_mask_permutexvar_ps(zmm[5], mask2,idx3,zmm[4]);
                zmm[27] = _mm512_mask_permutexvar_ps(zmm[7], mask2,idx3,zmm[6]);
                zmm[28] = _mm512_mask_permutexvar_ps(zmm[9], mask2,idx3,zmm[8]);
                zmm[29] = _mm512_mask_permutexvar_ps(zmm[11], mask2,idx3,zmm[10]);
                zmm[30] = _mm512_mask_permutexvar_ps(zmm[13], mask2,idx3,zmm[12]);
                zmm[31] = _mm512_mask_permutexvar_ps(zmm[15], mask2,idx3,zmm[14]);


                zmm[0] = _mm512_set1_ps(*(bk + 0));
                zmm[1] = _mm512_mul_ps(zmm[16],zmm[0]);
                res = _mm512_add_ps(zmm[1],res);

                zmm[0] = _mm512_set1_ps(*(bk + 1));
                zmm[1] = _mm512_mul_ps(zmm[17],zmm[0]);
                res = _mm512_add_ps(zmm[1],res);

                zmm[0] = _mm512_set1_ps(*(bk + 2));
                zmm[1] = _mm512_mul_ps(zmm[18],zmm[0]);
                res = _mm512_add_ps(zmm[1],res);

                zmm[0] = _mm512_set1_ps(*(bk + 3));
                zmm[1] = _mm512_mul_ps(zmm[19],zmm[0]);
                res = _mm512_add_ps(zmm[1],res);

                zmm[0] = _mm512_set1_ps(*(bk + 4));
                zmm[1] = _mm512_mul_ps(zmm[20],zmm[0]);
                res = _mm512_add_ps(zmm[1],res);

                zmm[0] = _mm512_set1_ps(*(bk + 5));
                zmm[1] = _mm512_mul_ps(zmm[21],zmm[0]);
                res = _mm512_add_ps(zmm[1],res);

                zmm[0] = _mm512_set1_ps(*(bk + 6));
                zmm[1] = _mm512_mul_ps(zmm[22],zmm[0]);
                res = _mm512_add_ps(zmm[1],res);

                zmm[0] = _mm512_set1_ps(*(bk + 7));
                zmm[1] = _mm512_mul_ps(zmm[23],zmm[0]);
                res = _mm512_add_ps(zmm[1],res);

                zmm[0] = _mm512_set1_ps(*(bk + 8));
                zmm[1] = _mm512_mul_ps(zmm[24],zmm[0]);
                res = _mm512_add_ps(zmm[1],res);

                zmm[0] = _mm512_set1_ps(*(bk + 9));
                zmm[1] = _mm512_mul_ps(zmm[25],zmm[0]);
                res = _mm512_add_ps(zmm[1],res);

                zmm[0] = _mm512_set1_ps(*(bk + 10));
                zmm[1] = _mm512_mul_ps(zmm[26],zmm[0]);
                res = _mm512_add_ps(zmm[1],res);

                zmm[0] = _mm512_set1_ps(*(bk + 11));
                zmm[1] = _mm512_mul_ps(zmm[27],zmm[0]);
                res = _mm512_add_ps(zmm[1],res);

                zmm[0] = _mm512_set1_ps(*(bk + 12));
                zmm[1] = _mm512_mul_ps(zmm[28],zmm[0]);
                res = _mm512_add_ps(zmm[1],res);

                zmm[0] = _mm512_set1_ps(*(bk + 13));
                zmm[1] = _mm512_mul_ps(zmm[29],zmm[0]);
                res = _mm512_add_ps(zmm[1],res);

                zmm[0] = _mm512_set1_ps(*(bk + 14));
                zmm[1] = _mm512_mul_ps(zmm[30],zmm[0]);
                res = _mm512_add_ps(zmm[1],res);

                zmm[0] = _mm512_set1_ps(*(bk + 15));
                zmm[1] = _mm512_mul_ps(zmm[31],zmm[0]);
                res = _mm512_add_ps(zmm[1],res);

                ak += wa[1] * 16;
                bk += 16;
            }

            k_iter = k_rem / 8;
            k_rem = k_rem % 8;

            if( k_iter ){
                
                //TODO: implement avx512 8x8

                ak += wa[1] * 8;
                bk += 8;
            }

            k_iter = k_rem / 4;
            k_rem = k_rem % 4;

            if( k_iter ){
                
                //TODO: implement avx512 4x4

                ak += wa[1] * 4;
                bk += 4;
            }


            _mm512_storeu_ps(ck,res);

            float sum = 0;
            #pragma omp simd reduction(+:sum)
            for(auto i = 0ul; i < M; ++i){
                auto aj = ak;
                auto bj = bk;
                auto cj = ck;
                for( auto j = 0ul; j < k_rem; ++j ){
                    sum += *aj * *bj;
                    ++aj;
                    ++bj;
                }
                *ck += sum;
                sum = 0;
                ak += wa[0];
                ++ck;
            }

        }
    };

    template<>
    struct kernel_helper<double,row_major>{
        static constexpr std::size_t M = 8;
        static constexpr std::size_t K = 8;

        template<typename SizeType>
        inline void operator()(double* c, SizeType const* nc, SizeType const* wc,
            double const* a, SizeType const* na, SizeType const* wa,
            double const* b, SizeType const* nb, SizeType const* wb
        ) const noexcept{
            static_assert(
                detail::wrap_cond_v< detail::simd_config::avx512f , SizeType> ,
                "Your processor does not support AVX512F instruction set"
            );

            __m512d zmm[17], res;

            auto k_iter = na[1] / K;
            auto k_rem = na[1] % K;

            auto ak = a;
            auto bk = b;
            auto ck = c;

            res = _mm512_loadu_pd(ck);

            while(k_iter--){
                zmm[0] = _mm512_loadu_pd(ak);
                zmm[1] = _mm512_loadu_pd(ak + wa[1]);
                zmm[2] = _mm512_loadu_pd(ak + wa[1] * 2);
                zmm[3] = _mm512_loadu_pd(ak + wa[1] * 3);
                zmm[4] = _mm512_loadu_pd(ak + wa[1] * 4);
                zmm[5] = _mm512_loadu_pd(ak + wa[1] * 5);
                zmm[6] = _mm512_loadu_pd(ak + wa[1] * 6);
                zmm[7] = _mm512_loadu_pd(ak + wa[1] * 7);

                zmm[16] = _mm512_set1_pd(*bk);
                zmm[16] = _mm512_mul_pd(zmm[16],zmm[0]);
                res = _mm512_add_pd(zmm[16],res);

                zmm[16] = _mm512_set1_pd(*( bk + 1) );
                zmm[16] = _mm512_mul_pd(zmm[16],zmm[1]);
                res = _mm512_add_pd(zmm[16],res);

                zmm[16] = _mm512_set1_pd(*( bk + 2) );
                zmm[16] = _mm512_mul_pd(zmm[16],zmm[2]);
                res = _mm512_add_pd(zmm[16],res);

                zmm[16] = _mm512_set1_pd(*( bk + 3) );
                zmm[16] = _mm512_mul_pd(zmm[16],zmm[3]);
                res = _mm512_add_pd(zmm[16],res);

                zmm[16] = _mm512_set1_pd(*( bk + 4) );
                zmm[16] = _mm512_mul_pd(zmm[16],zmm[4]);
                res = _mm512_add_pd(zmm[16],res);

                zmm[16] = _mm512_set1_pd(*( bk + 5) );
                zmm[16] = _mm512_mul_pd(zmm[16],zmm[5]);
                res = _mm512_add_pd(zmm[16],res);

                zmm[16] = _mm512_set1_pd(*( bk + 6) );
                zmm[16] = _mm512_mul_pd(zmm[16],zmm[6]);
                res = _mm512_add_pd(zmm[16],res);

                zmm[16] = _mm512_set1_pd(*( bk + 7) );
                zmm[16] = _mm512_mul_pd(zmm[16],zmm[7]);
                res = _mm512_add_pd(zmm[16],res);

                ak += wa[1] * K;
                bk += K;
            }

            k_iter = k_rem / 4;
            k_rem = k_rem % 4;

            if( k_iter ){
                zmm[0] = _mm512_loadu_pd(ak);
                zmm[1] = _mm512_loadu_pd(ak + wa[1]);
                zmm[2] = _mm512_loadu_pd(ak + wa[1] * 2);
                zmm[3] = _mm512_loadu_pd(ak + wa[1] * 3);

                zmm[16] = _mm512_set1_pd(*bk);
                zmm[16] = _mm512_mul_pd(zmm[16],zmm[0]);
                res = _mm512_add_pd(zmm[16],res);

                zmm[16] = _mm512_set1_pd(*( bk + 1) );
                zmm[16] = _mm512_mul_pd(zmm[16],zmm[1]);
                res = _mm512_add_pd(zmm[16],res);

                zmm[16] = _mm512_set1_pd(*( bk + 2) );
                zmm[16] = _mm512_mul_pd(zmm[16],zmm[2]);
                res = _mm512_add_pd(zmm[16],res);

                zmm[16] = _mm512_set1_pd(*( bk + 3) );
                zmm[16] = _mm512_mul_pd(zmm[16],zmm[3]);
                res = _mm512_add_pd(zmm[16],res);

                ak += wa[1] * 4;
                bk += 4;
            }

            k_iter = k_rem / 2;
            k_rem = k_rem % 2;

            if( k_iter ){
                zmm[0] = _mm512_loadu_pd(ak);
                zmm[1] = _mm512_loadu_pd(ak + wa[1]);

                zmm[16] = _mm512_set1_pd(*bk);
                zmm[16] = _mm512_mul_pd(zmm[16],zmm[0]);
                res = _mm512_add_pd(zmm[16],res);

                zmm[16] = _mm512_set1_pd(*( bk + 1) );
                zmm[16] = _mm512_mul_pd(zmm[16],zmm[1]);
                res = _mm512_add_pd(zmm[16],res);

                ak += wa[1] * 2;
                bk += 2;
            }

            if( k_rem == 1 ){
                zmm[0] = _mm512_loadu_pd(ak);
                
                zmm[16] = _mm512_set1_pd(*bk);
                zmm[16] = _mm512_mul_pd(zmm[16],zmm[0]);
                res = _mm512_add_pd(zmm[16],res);

            }

            _mm512_storeu_pd(ck,res);

        }
    };

} // namespace tlib::simd::x86::



// row_major
namespace tlib::simd::x86{
    
    template<>
    struct kernel<row_major>{

        template<typename ValueType, typename SizeType>
        inline void operator()(ValueType* c, SizeType const* nc, SizeType const* wc,
            ValueType const* a, SizeType const* na, SizeType const* wa,
            ValueType const* b, SizeType const* nb, SizeType const* wb
        ) const noexcept{

            auto ai = a;
            auto bi = b;
            auto ci = c;

            auto i_iter = 0ul;
            auto i_rem = 0ul;
            auto m = na[1];
            
            if constexpr ( detail::simd_config::avx512f ){
                using kernel_type512 = avx512::kernel_helper<ValueType,row_major>;
                constexpr auto M = kernel_type512::M;
                
                auto const wa_0 = wa[0] * M; 
                auto const wc_0 = wc[0] * M;

                i_iter = m / M;
                i_rem = m % M;

                while(i_iter--){

                    SizeType const nta[] = { M, na[0] };
                    SizeType const ntb[] = { na[0], 1 };
                    SizeType const ntc[] = { M, 1 };

                    kernel_type512{}(
                        ci, ntc, wc,
                        ai, nta, wa,
                        bi, ntb, wb
                    );

                    ai += wa_0;
                    ci += wc_0;
                }

                m = i_rem;

            }

            if constexpr ( detail::simd_config::avx && detail::simd_config::avx2 ){
                using kernel_type256 = avx256::kernel_helper<ValueType,row_major>;
                constexpr auto M = kernel_type256::M;
                auto const wa_0 = wa[0] * M; 
                auto const wc_0 = wc[0] * M;

                i_iter = m / M;
                i_rem = m % M;

                while(i_iter--){
                    SizeType const nta[] = { M, na[0] };
                    SizeType const ntb[] = { na[0], 1 };
                    SizeType const ntc[] = { M, 1 };

                    kernel_type256{}(
                        ci, ntc, wc,
                        ai, nta, wa,
                        bi, ntb, wb
                    );

                    ai += wa_0;
                    ci += wc_0;
                }


                m = i_rem;
            }

            if constexpr ( 
                detail::simd_config::sse && detail::simd_config::sse2 &&
                detail::simd_config::sse3 && detail::simd_config::sse4_1
            ){
                using kernel_types128 = sse::kernel_helper<ValueType,row_major>;
                constexpr auto M = kernel_types128::M;
                
                auto const wa_0 = wa[0] * M; 
                auto const wc_0 = wc[0] * M;
                i_iter = m / M;
                i_rem = m % M;

                while(i_iter--){

                    SizeType const nta[] = { M, na[0] };
                    SizeType const ntb[] = { na[0], 1 };
                    SizeType const ntc[] = { M, 1 };

                    kernel_types128{}(
                        ci, ntc, wc,
                        ai, nta, wa,
                        bi, ntb, wb
                    );

                    ai += wa_0;
                    ci += wc_0;
                }
            }
            
            if ( i_rem ){
                SizeType const nta[] = { i_rem, na[0]};

                SizeType const ntb[] = { na[0], 1};

                SizeType const ntc[] = { i_rem , 1};

                auto aj = ai;
                auto bj = bi;
                auto cj = ci;

                helper_mxn(
                    cj, ntc, wc,
                    aj, nta, wa,
                    bj, ntb, wb
                );
            }

        }

    private:

        template<typename ValueType, typename SizeType>
        inline void helper_mxn(ValueType* c, SizeType const* nc, SizeType const* wc,
            ValueType const* a, SizeType const* na, SizeType const* wa,
            ValueType const* b, SizeType const* nb, SizeType const* wb
        ) const noexcept{
            
            auto i_rem = na[0];

            auto ak = a;
            auto bk = b;
            auto ck = c;

            if( i_rem == 1 ){
                
                auto sum = ValueType{};
                for( auto k = 0ul; k < na[1]; ++k ){
                    sum += *( ak + wa[1] * k ) * *(bk + k);
                }
                *ck += sum;

            }else if ( i_rem == 2 ){
                
                auto sum = ValueType{};
                #pragma omp simd reduction(+:sum)
                for( auto i = 0ul; i < 2; ++i ){
                    for( auto k = 0ul; k < na[1]; ++k ){
                        sum += *( ak + wa[1] * k + wa[0] * i ) * *(bk + k);
                    }
                    *ck += sum;
                    ++ck;
                    sum = ValueType{};
                }

            }else if( i_rem == 3 ){
                
                auto sum = ValueType{};
                #pragma omp simd reduction(+:sum)
                for( auto i = 0ul; i < 3; ++i ){
                    for( auto k = 0ul; k < na[1]; ++k ){
                        sum += *( ak + wa[1] * k + wa[0] * i ) * *(bk + k);
                    }
                    *ck += sum;
                    ++ck;
                    sum = ValueType{};
                }

            }

        }

    };

} // namespace tlib::simd

#endif // TLIB_DETAIL_KERNEL_X86_ROW_H
