#ifndef TLIB_DETAIL_KERNEL_PARTITION_H
#define TLIB_DETAIL_KERNEL_PARTITION_H

#include <cstddef>
#include "layout.h"
#include <type_traits>
#include "simd_support.h"

namespace tlib::simd::x86{
    
    template<typename,typename>
    struct partition;

} // namespace tlib::simd

namespace tlib::simd::x86{
    
    template<typename T, typename F>
    struct partition{
        using size_type = std::size_t;

        static constexpr auto min_size = 256 / (8 * sizeof(T));

        constexpr partition() = default;
        
        constexpr partition(size_type m, size_type k, size_type n)
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
            auto div = m / ( min_size );
            auto temp = std::max( div - div % min_size, min_size );
            m_m = temp;
            m_k = temp;

            if( m > k ){
                if( m > 2000 ){
                    m_k = min_size;
                }else{
                    m_m = 96ul;
                }
            }else{
                if( k < 2000 ){
                    m_k = min_size;
                }else{
                    m_m = 96ul;
                }
            }


        }

        inline constexpr void calc_row(size_type m, size_type k, size_type n) noexcept{
            auto div_m = m / ( min_size );
            auto div_k = k / ( min_size );
            auto temp_k = std::max( div_k - div_k % min_size, min_size );
            auto temp_m = std::max( div_m - div_m % min_size, min_size );
            m_m = temp_m;
            m_k = temp_k;

            if( m > k && m > 2000 ){
                m_k = min_size;
            }else if ( m < k && k < 2000){
                m_k = min_size;
            }
        }

    private:
        size_type m_m{8};
        size_type m_k{8};
        size_type m_n{1};
    };

} // namespace tlib::simd



#endif // TLIB_DETAIL_KERNEL_PARTITION_H
