#ifndef TLIB_DETAIL_KERNEL_X86_COL_H
#define TLIB_DETAIL_KERNEL_X86_COL_H

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
    struct kernel_helper<float,col_major>{
        
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

            detail::VFloat r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, res;

            auto k_iter = na[1] / K;
            auto k_rem = na[1] % K;

            auto ak = a;
            auto bk = b;
            auto ck = c;

            res.y = _mm256_loadu_ps(ck);

            while(k_iter--){
                r0.y = _mm256_loadu_ps(ak);
                r1.y = _mm256_loadu_ps(ak + wa[1]);
                r2.y = _mm256_loadu_ps(ak + wa[1] * 2);
                r3.y = _mm256_loadu_ps(ak + wa[1] * 3);
                r4.y = _mm256_loadu_ps(ak + wa[1] * 4);
                r5.y = _mm256_loadu_ps(ak + wa[1] * 5);
                r6.y = _mm256_loadu_ps(ak + wa[1] * 6);
                r7.y = _mm256_loadu_ps(ak + wa[1] * 7);

                r8.y = _mm256_loadu_ps(bk);

                r9.y = _mm256_permute2f128_ps(r8.y, r8.y, 0x0);
                r10.y = _mm256_permute2f128_ps(r8.y, r8.y, 0x11);

                r11.y = _mm256_permute_ps(r9.y, _MM_SHUFFLE(0,0,0,0));
                detail::fmadd(r0,r11,res);
                
                r11.y = _mm256_permute_ps(r9.y, _MM_SHUFFLE(1,1,1,1));
                detail::fmadd(r1,r11,res);
                
                r11.y = _mm256_permute_ps(r9.y, _MM_SHUFFLE(2,2,2,2));
                detail::fmadd(r2,r11,res);
                
                r11.y = _mm256_permute_ps(r9.y, _MM_SHUFFLE(3,3,3,3));
                detail::fmadd(r3,r11,res);

                r11.y = _mm256_permute_ps(r10.y, _MM_SHUFFLE(0,0,0,0));
                detail::fmadd(r4,r11,res);
                
                r11.y = _mm256_permute_ps(r10.y, _MM_SHUFFLE(1,1,1,1));
                detail::fmadd(r5,r11,res);
                
                r11.y = _mm256_permute_ps(r10.y, _MM_SHUFFLE(2,2,2,2));
                detail::fmadd(r6,r11,res);
                
                r11.y = _mm256_permute_ps(r10.y, _MM_SHUFFLE(3,3,3,3));
                detail::fmadd(r7,r11,res);

                ak += wa[1] * K;
                bk += K;
            }

            k_iter = k_rem / 4;
            k_rem = k_rem % 4;

            if( k_iter ){

                r0.y = _mm256_loadu_ps(ak);
                r1.y = _mm256_loadu_ps(ak + wa[1]);
                r2.y = _mm256_loadu_ps(ak + wa[1] * 2);
                r3.y = _mm256_loadu_ps(ak + wa[1] * 3);

                r4.y = _mm256_broadcast_ss(bk);
                detail::fmadd(r0,r4,res);

                r4.y = _mm256_broadcast_ss(bk + 1);
                detail::fmadd(r1,r4,res);

                r4.y = _mm256_broadcast_ss(bk + 2);
                detail::fmadd(r2,r4,res);

                r4.y = _mm256_broadcast_ss(bk + 3);
                detail::fmadd(r3,r4,res);

                ak += wa[1] * 4;
                bk += 4;

            }

            if( k_rem == 1 ){

                r0.y = _mm256_loadu_ps(ak);

                r4.y = _mm256_broadcast_ss(bk);
                detail::fmadd(r0,r4,res);

            }else if( k_rem == 2 ){

                r0.y = _mm256_loadu_ps(ak);
                r1.y = _mm256_loadu_ps(ak + wa[1]);
                
                r4.y = _mm256_broadcast_ss(bk);
                detail::fmadd(r0,r4,res);

                r4.y = _mm256_broadcast_ss(bk + 1);
                detail::fmadd(r1,r4,res);

            }else if ( k_rem == 3 ){
                r0.y = _mm256_loadu_ps(ak);
                r1.y = _mm256_loadu_ps(ak + wa[1]);
                r2.y = _mm256_loadu_ps(ak + wa[1] * 2);

                r4.y = _mm256_broadcast_ss(bk);
                detail::fmadd(r0,r4,res);

                r4.y = _mm256_broadcast_ss(bk + 1);
                detail::fmadd(r1,r4,res);

                r4.y = _mm256_broadcast_ss(bk + 2);
                detail::fmadd(r2,r4,res);
            }

            _mm256_storeu_ps(ck, res.y);
        }

    };
    
    template<>
    struct kernel_helper<double,col_major>{
        
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

            detail::VDouble r0, r1, r2, r3, r4, r8, r9, r10, r11, res;

            auto k_iter = na[1] / K;
            auto k_rem = na[1] % K;

            auto ak = a;
            auto bk = b;
            auto ck = c;

            res.y = _mm256_loadu_pd(ck);

            while(k_iter--){
                r0.y = _mm256_loadu_pd(ak);
                r1.y = _mm256_loadu_pd(ak + wa[1]);
                r2.y = _mm256_loadu_pd(ak + wa[1] * 2);
                r3.y = _mm256_loadu_pd(ak + wa[1] * 3);

                r11.y = _mm256_broadcast_sd(bk);
                detail::fmadd(r0,r11,res);
                
                r11.y = _mm256_broadcast_sd(bk + 1);
                detail::fmadd(r1,r11,res);
                
                r11.y = _mm256_broadcast_sd(bk + 2);
                detail::fmadd(r2,r11,res);
                
                r11.y = _mm256_broadcast_sd(bk + 3);
                detail::fmadd(r3,r11,res);

                ak += wa[1] * K;
                bk += K;
            }

            k_iter = k_rem / 2;
            k_rem = k_rem % 2;

            if( k_iter ){

                r0.y = _mm256_loadu_pd(ak);
                r1.y = _mm256_loadu_pd(ak + wa[1]);

                r4.y = _mm256_broadcast_sd(bk);
                detail::fmadd(r0,r4,res);

                r4.y = _mm256_broadcast_sd(bk + 1);
                detail::fmadd(r1,r4,res);

                ak += wa[1] * 2;
                bk += 2;

            }

            if( k_rem == 1 ){

                r0.y = _mm256_loadu_pd(ak);

                r4.y = _mm256_broadcast_sd(bk);
                detail::fmadd(r0,r4,res);

            }

            _mm256_storeu_pd(ck, res.y);
        }

    };
    
} // namespace tlib::simd::x86::avx256

namespace tlib::simd::x86::sse{
    template<>
    struct kernel_helper<float,col_major>{
        
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
                r1.x[0] = _mm_loadu_ps(ak + wa[1]);
                r2.x[0] = _mm_loadu_ps(ak + wa[1] * 2);
                r3.x[0] = _mm_loadu_ps(ak + wa[1] * 3);
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

            if( k_rem == 1 ){

                r0.x[0] = _mm_loadu_ps(ak);

                r4.x[0] = _mm_broadcast_ss(bk);
                detail::fmadd(r0,r4,res,0);

            }else if( k_rem == 2 ){

                r0.x[0] = _mm_loadu_ps(ak);
                r1.x[0] = _mm_loadu_ps(ak + wa[1]);
                
                r4.x[0] = _mm_broadcast_ss(bk);
                detail::fmadd(r0,r4,res,0);

                r4.x[0] = _mm_broadcast_ss(bk + 1);
                detail::fmadd(r1,r4,res,0);

            }else if ( k_rem == 3 ){
                r0.x[0] = _mm_loadu_ps(ak);
                r1.x[0] = _mm_loadu_ps(ak + wa[1]);
                r2.x[0] = _mm_loadu_ps(ak + wa[1] * 2);

                r4.x[0] = _mm_broadcast_ss(bk);
                detail::fmadd(r0,r4,res,0);

                r4.x[0] = _mm_broadcast_ss(bk + 1);
                detail::fmadd(r1,r4,res,0);

                r4.x[0] = _mm_broadcast_ss(bk + 2);
                detail::fmadd(r2,r4,res,0);
            }

            _mm_storeu_ps(ck, res.x[0]);
        }

    };

    template<>
    struct kernel_helper<double,col_major>{
        
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

            auto k_iter = na[1] / K;
            auto k_rem = na[1] % K;

            auto ak = a;
            auto bk = b;
            auto ck = c;

            res.x[0] = _mm_loadu_pd(ck);

            while(k_iter--){
                r0.x[0] = _mm_loadu_pd(ak);
                r1.x[0] = _mm_loadu_pd(ak + wa[1]);
                
                r8.x[0] = _mm_loadu_pd(bk);

                r11.x[0] = _mm_permute_pd(r8.x[0], _MM_SHUFFLE2(0,0));
                detail::fmadd(r0,r11,res,0);
                
                r11.x[0] = _mm_permute_pd(r8.x[0], _MM_SHUFFLE2(1,1));
                detail::fmadd(r1,r11,res,0);

                ak += wa[1] * K;
                bk += K;
            }

            if( k_rem == 1 ){

                r0.x[0] = _mm_loadu_pd(ak);

                r4.x[0] = _mm_set1_pd(*bk);
                detail::fmadd(r0,r4,res,0);

            }

            _mm_storeu_pd(ck, res.x[0]);
        }

    };

} // namespace tlib::simd::x86::sse

namespace tlib::simd::x86::avx512{
    
    template<>
    struct kernel_helper<float,col_major>{
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

            __m512 zmm[17], res;

            auto k_iter = na[1] / K;
            auto k_rem = na[1] % K;

            auto ak = a;
            auto bk = b;
            auto ck = c;

            res = _mm512_loadu_ps(ck);

            while(k_iter--){
                zmm[0] = _mm512_loadu_ps(ak);
                zmm[1] = _mm512_loadu_ps(ak + wa[1]);
                zmm[2] = _mm512_loadu_ps(ak + wa[1] * 2);
                zmm[3] = _mm512_loadu_ps(ak + wa[1] * 3);
                zmm[4] = _mm512_loadu_ps(ak + wa[1] * 4);
                zmm[5] = _mm512_loadu_ps(ak + wa[1] * 5);
                zmm[6] = _mm512_loadu_ps(ak + wa[1] * 6);
                zmm[7] = _mm512_loadu_ps(ak + wa[1] * 7);
                zmm[8] = _mm512_loadu_ps(ak + wa[1] * 8);
                zmm[9] = _mm512_loadu_ps(ak + wa[1] * 9);
                zmm[10] = _mm512_loadu_ps(ak + wa[1] * 10);
                zmm[11] = _mm512_loadu_ps(ak + wa[1] * 11);
                zmm[12] = _mm512_loadu_ps(ak + wa[1] * 12);
                zmm[13] = _mm512_loadu_ps(ak + wa[1] * 13);
                zmm[14] = _mm512_loadu_ps(ak + wa[1] * 14);
                zmm[15] = _mm512_loadu_ps(ak + wa[1] * 15);

                zmm[16] = _mm512_set1_ps(*bk);
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[0]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 1) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[1]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 2) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[2]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 3) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[3]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 4) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[4]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 5) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[5]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 6) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[6]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 7) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[7]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 8) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[8]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 9) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[9]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 10) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[10]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 11) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[11]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 12) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[12]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 13) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[13]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 14) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[14]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 15) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[15]);
                res = _mm512_add_ps(zmm[16],res);

                ak += wa[1] * 16;
                bk += 16;
            }

            k_iter = k_rem / 8;
            k_rem = k_rem % 8;

            if( k_iter ){
                zmm[0] = _mm512_loadu_ps(ak);
                zmm[1] = _mm512_loadu_ps(ak + wa[1]);
                zmm[2] = _mm512_loadu_ps(ak + wa[1] * 2);
                zmm[3] = _mm512_loadu_ps(ak + wa[1] * 3);
                zmm[4] = _mm512_loadu_ps(ak + wa[1] * 4);
                zmm[5] = _mm512_loadu_ps(ak + wa[1] * 5);
                zmm[6] = _mm512_loadu_ps(ak + wa[1] * 6);
                zmm[7] = _mm512_loadu_ps(ak + wa[1] * 7);

                zmm[16] = _mm512_set1_ps(*bk);
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[0]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 1) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[1]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 2) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[2]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 3) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[3]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 4) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[4]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 5) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[5]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 6) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[6]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 7) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[7]);
                res = _mm512_add_ps(zmm[16],res);

                ak += wa[1] * 8;
                bk += 8;
            }

            k_iter = k_rem / 4;
            k_rem = k_rem % 4;

            if( k_iter ){
                zmm[0] = _mm512_loadu_ps(ak);
                zmm[1] = _mm512_loadu_ps(ak + wa[1]);
                zmm[2] = _mm512_loadu_ps(ak + wa[1] * 2);
                zmm[3] = _mm512_loadu_ps(ak + wa[1] * 3);

                zmm[16] = _mm512_set1_ps(*bk);
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[0]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 1) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[1]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 2) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[2]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 3) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[3]);
                res = _mm512_add_ps(zmm[16],res);

                ak += wa[1] * 4;
                bk += 4;
            }

            if( k_rem == 1 ){
                zmm[0] = _mm512_loadu_ps(ak);
                
                zmm[16] = _mm512_set1_ps(*bk);
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[0]);
                res = _mm512_add_ps(zmm[16],res);

            }else if ( k_rem == 2 ){

                zmm[0] = _mm512_loadu_ps(ak);
                zmm[1] = _mm512_loadu_ps(ak + wa[1]);
                
                zmm[16] = _mm512_set1_ps(*bk);
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[0]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 1) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[1]);
                res = _mm512_add_ps(zmm[16],res);

            }else if ( k_rem == 3 ){
                zmm[0] = _mm512_loadu_ps(ak);
                zmm[1] = _mm512_loadu_ps(ak + wa[1]);
                zmm[2] = _mm512_loadu_ps(ak + wa[1] * 2);

                zmm[16] = _mm512_set1_ps(*bk);
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[0]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 1) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[1]);
                res = _mm512_add_ps(zmm[16],res);

                zmm[16] = _mm512_set1_ps(*( bk + 2) );
                zmm[16] = _mm512_mul_ps(zmm[16],zmm[2]);
                res = _mm512_add_ps(zmm[16],res);
            }

            _mm512_storeu_ps(ck,res);

        }
    };

    template<>
    struct kernel_helper<double,col_major>{
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



// col_major
namespace tlib::simd::x86{
    
    template<>
    struct kernel<col_major>{

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
            auto m = na[0];

            if constexpr ( detail::simd_config::avx512f ){
                using kernel_type512 = avx512::kernel_helper<ValueType,col_major>;
                constexpr auto M = kernel_type512::M;
                
                auto const wa_0 = wa[0] * M; 
                auto const wc_0 = wc[0] * M;

                i_iter = m / M;
                i_rem = m % M;

                while(i_iter--){

                    SizeType const nta[] = { M, na[1] };
                    SizeType const ntb[] = { na[1], 1 };
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
                using kernel_type256 = avx256::kernel_helper<ValueType,col_major>;
                constexpr auto M = kernel_type256::M;
                auto const wa_0 = wa[0] * M; 
                auto const wc_0 = wc[0] * M;

                i_iter = m / M;
                i_rem = m % M;

                while(i_iter--){

                    SizeType const nta[] = { M, na[1] };
                    SizeType const ntb[] = { na[1], 1 };
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
                using kernel_types128 = sse::kernel_helper<ValueType,col_major>;
                constexpr auto M = kernel_types128::M;
                
                auto const wa_0 = wa[0] * M; 
                auto const wc_0 = wc[0] * M;

                i_iter = m / M;
                i_rem = m % M;

                while(i_iter--){

                    SizeType const nta[] = { M, na[1] };
                    SizeType const ntb[] = { na[1], 1 };
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
                SizeType const nta[] = { i_rem, na[1] };
                SizeType const ntb[] = { na[1], 1 };
                SizeType const ntc[] = { i_rem, 1 };

                helper_mxn(
                    ci, ntc, wc,
                    ai, nta, wa,
                    bi, ntb, wb
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
                    ck += wc[0];
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
                    ck += wc[0];
                    sum = ValueType{};
                }

            }

        }

    };

} // namespace tlib::simd

#endif // TLIB_DETAIL_KERNEL_X86_COL_H
