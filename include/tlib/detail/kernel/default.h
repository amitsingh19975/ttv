#ifndef TLIB_DETAIL_KERNEL_DEFAULT_H
#define TLIB_DETAIL_KERNEL_DEFAULT_H

#include <cstddef>
#include "layout.h"

namespace tlib::simd{
    
    template<typename>
    struct kernel;


} // namespace tlib::simd


namespace tlib::simd{
    
    template<>
    struct kernel<col_major>{

        static constexpr std::size_t const M = 8;
        static constexpr std::size_t const K = 8;

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
            
            auto const wa_0 = wa[0] * M; 
            auto const wc_0 = wc[0] * M;

            i_iter = m / M;
            i_rem = m % M;

            while(i_iter--){

                SizeType const nta[] = { M, na[1] };
                SizeType const ntb[] = { na[1], 1 };
                SizeType const ntc[] = { M, 1 };

                kernel_helper(
                    ci, ntc, wc,
                    ai, nta, wa,
                    bi, ntb, wb
                );

                ai += wa_0;
                ci += wc_0;
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
        inline void kernel_helper(ValueType* c, SizeType const* nc, SizeType const* wc,
            ValueType const* a, SizeType const* na, SizeType const* wa,
            ValueType const* b, SizeType const* nb, SizeType const* wb
        ) const noexcept
        {
            auto ak = a;
            auto bk = b;
            auto ck = c;

            auto k_iter = na[1] / K;
            auto k_rem = na[1] % K;

            while(k_iter--){
                for( auto k = 0ul; k < K; ++k){
                    auto ai = ak;
                    auto bi = bk;
                    auto ci = ck;
                    #pragma omp simd aligned(ai,bi,ci:32) linear(bi,ci:1)
                    for(auto i = 0ul; i < M; ++i){
                        *ci += *ai * *bi;
                        ++ai;
                        ++ci;
                    }
                    ak += wa[1];
                    ++bk;
                }
            }

            for( auto k = 0ul; k < k_rem; ++k){
                auto ai = ak;
                auto bi = bk;
                auto ci = ck;
                #pragma omp simd aligned(ai,bi,ci:32) linear(bi,ci:1)
                for(auto i = 0ul; i < M; ++i){
                    *ci += *ai * *bi;
                    ++ai;
                    ++ci;
                }
                ak += wa[1];
                ++bk;
            }

        }

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

#endif // TLIB_DETAIL_KERNEL_DEFAULT_H
