#ifndef TLIB_DETAIL_KERNEL_PARTITION_H
#define TLIB_DETAIL_KERNEL_PARTITION_H

#include <cstddef>
#include "layout.h"
#include <type_traits>

namespace tlib::simd{
    
    template<typename>
    struct x86_partition;

} // namespace tlib::simd

namespace tlib::simd{
    
    template<typename F>
    struct x86_partition{
        using size_type = std::size_t;
        constexpr x86_partition() = default;
        
        constexpr x86_partition(size_type m, size_type k, size_type n)
        {
            if constexpr( std::is_same_v<F,col_major> ){
                calc_col(m,n,k);
            }else{
                calc_row(m,n,k);
            }
        }

        inline constexpr void operator()(size_type m, size_type k, size_type n) noexcept{
            if constexpr( std::is_same_v<F,col_major> ){
                calc_col(m,n,k);
            }else{
                calc_row(m,n,k);
            }
        }

        inline constexpr size_type M() const noexcept{
            return m_m;
        }

        inline constexpr size_type N() const noexcept{
            return m_n;
        }

        inline constexpr size_type K() const noexcept{
            return m_k;
        }

    private:

        inline constexpr void calc_col(size_type m, size_type k, size_type n) noexcept{
            if( m > 2000 ){
                auto div_m = m / 4;
                m_m = std::max( div_m - div_m % 8, 8ul );
                if( k > 2000 ){
                    auto div_k = m / 4;
                    m_k = std::max( div_k - div_k % 8, 8ul );
                }else{
                    m_k = 8ul;
                }
            }else{
                auto div_k = m / 4;
                m_k = std::max( div_k - div_k % 8, 8ul );
                m_m = 88ul;
            }
        }

        inline constexpr void calc_row(size_type m, size_type k, size_type n) noexcept{

        }

    private:
        size_type m_m{8};
        size_type m_k{8};
        size_type m_n{1};
    };

} // namespace tlib::simd



#endif // TLIB_DETAIL_KERNEL_PARTITION_H
