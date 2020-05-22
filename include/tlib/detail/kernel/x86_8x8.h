#ifndef TLIB_DETAIL_KERNEL_X86_8x8_H
#define TLIB_DETAIL_KERNEL_X86_8x8_H

#include <cstddef>
#include "simd.h"
#include "layout.h"

namespace tlib::simd{
    
    template<typename>
    struct x86_kernel{};

} // namespace tlib::simd

// col_major
namespace tlib::simd{
    
    template<>
    struct x86_kernel<col_major>{
        
        static constexpr std::size_t M = 8;
        static constexpr std::size_t K = 8;

        template<typename SizeType>
        inline void operator()(float* c, SizeType const* nc, SizeType const* wc,
            float const* a, SizeType const* na, SizeType const* wa,
            float const* b, SizeType const* nb, SizeType const* wb
        ) const noexcept{

            static_assert(detail::simd_config::avx2 && detail::simd_config::avx, "your processor is not supported");

            auto ai = a;
            auto bi = b;
            auto ci = c;

            auto const wa_0 = wa[0] * M; 
            auto const wc_0 = wc[0] * M;

            auto i_iter = na[0] / M;
            auto i_rem = na[0] % M;

            while(i_iter){

                SizeType const nta[] = { M, na[1] };
                SizeType const ntb[] = { na[1], 1 };
                SizeType const ntc[] = { M, 1 };

                helper_8xn(
                    ci, ntc, wc,
                    ai, nta, wa,
                    bi, ntb, wb
                );

                ai += wa_0;
                ci += wc_0;
                --i_iter;
            }

        }

    private:

        template<typename SizeType>
        inline void helper_8xn(float* c, SizeType const* nc, SizeType const* wc,
            float const* a, SizeType const* na, SizeType const* wa,
            float const* b, SizeType const* nb, SizeType const* wb
        ) const noexcept{
            detail::VFloat r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, res;

            auto k_iter = na[1] / K;
            auto k_rem = na[1] % K;

            auto ak = a;
            auto bk = b;
            auto ck = c;

            res.y = _mm256_loadu_ps(ck);

            while(k_iter){
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

                ak += wa[1] * 8;
                bk += 8;
                --k_iter;
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

} // namespace tlib::simd



#endif // TLIB_DETAIL_KERNEL_X86_8x8_H
