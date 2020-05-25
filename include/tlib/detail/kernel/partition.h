#ifndef TLIB_DETAIL_KERNEL_PARTITION_H
#define TLIB_DETAIL_KERNEL_PARTITION_H

#include <cstddef>
#include "layout.h"
#include <type_traits>
#include "simd_support.h"
#include <cmath>

namespace tlib::simd::x86{
    
    template<typename,typename>
    struct partition;

} // namespace tlib::simd

namespace tlib::simd::x86{
    
    template<typename T, typename F>
    struct partition{
        using size_type = std::size_t;

        static constexpr auto const data_size = ( 8 * sizeof(T) );
        static constexpr auto const L1_cache = 32768 * 64 / data_size;

        constexpr partition() = default;
        
        constexpr partition(size_type m, size_type k, size_type n)
        {
            if constexpr( std::is_same_v<F,col_major> ){
                calc_col(m,k,n);
            }else{
                calc_row(m,k,n);
            }
        }

        inline constexpr void operator()(size_type m, size_type k, size_type n) noexcept{
            if constexpr( std::is_same_v<F,col_major> ){
                calc_col(m,k,n);
            }else{
                calc_row(m,k,n);
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

        inline constexpr size_t best_min_size(size_t sz) const noexcept{
            constexpr auto data_size = 8 * sizeof(T);
            constexpr auto d512 = 512ul / data_size;
            constexpr auto d256 = 256ul / data_size;
            constexpr auto d128 = 128ul / data_size;

            if constexpr( detail::simd_config::avx512f || 
                detail::simd_config::avx512bw ||
                detail::simd_config::avx512cd ||
                detail::simd_config::avx512dq ||
                detail::simd_config::avx512vl 
            ){
                if( sz < d512 ){
                    if constexpr ( detail::simd_config::avx ||
                        detail::simd_config::avx2
                    ){
                        return sz <= d256 ? d128 : d256;
                    }else{
                        return d128;
                    }
                }else{
                    return d512;
                }
            }else{
                if constexpr ( detail::simd_config::avx ||
                    detail::simd_config::avx2
                ){
                    return sz < d256 ? d128 : d256;
                }else{
                    return d128;
                }
            }
        }

        inline constexpr void calc_col(size_type m, size_type k, size_type n) noexcept{
            auto const k_min = best_min_size(k);
            auto const m_min = best_min_size(m);

            auto div_m = m / ( m_min );
            auto div_k = k / ( k_min );
            auto temp_k = std::max( div_k - div_k % k_min, k_min );
            auto temp_m = std::max( div_m - div_m % m_min, m_min );
            if( m * k > L1_cache ){
                m_m = temp_m;
                m_k = m > 2000ul ? m_min / 2ul : m_min;
            }else{
                std::size_t m_ltemp = std::sqrt( L1_cache );
                auto m_l = std::max( m_ltemp - (m_ltemp % m_min), temp_m);
                auto k_l = std::max( m_ltemp - (m_ltemp % k_min), temp_k);
                m_m = m_l;
                m_k = k_l;
            }
        }

        inline constexpr void calc_row(size_type m, size_type k, size_type n) noexcept{
            auto const k_min = best_min_size(k);
            auto const m_min = best_min_size(m);

            auto div_m = m / ( m_min );
            auto div_k = k/ ( k_min );
            auto temp_k = std::max( div_k - div_k % k_min, k_min );
            auto temp_m = std::max( div_m - div_m % m_min, m_min );
            std::size_t m_ltemp = std::sqrt( L1_cache );
            if( m * k > L1_cache ){
                m_m = temp_m;
                m_k = m > 2000ul ? m_min / 2ul : m_min;
            }else{
                auto m_l = std::max( m_ltemp - (m_ltemp % m_min), temp_m);
                auto k_l = std::max( m_ltemp - (m_ltemp % k_min), temp_k);
                m_m = m_l;
                m_k = k_l;
            }
        }

    private:
        size_type m_m{8};
        size_type m_k{8};
        size_type m_n{1};
    };

} // namespace tlib::simd



#endif // TLIB_DETAIL_KERNEL_PARTITION_H
