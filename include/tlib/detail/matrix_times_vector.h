#ifndef TLIB_DETAIL_MV_H
#define TLIB_DETAIL_MV_H

#include <cstddef>
#include <stdexcept>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <omp.h>


#include "tags.h"
#include "cases.h"
#include "kernel/x86.h"
#include "kernel/partition.h"


// <cblas.h>
#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

#ifdef USE_INTELBLAS
#include <mkl.h>
#endif


namespace tlib::detail {


/** \brief computes 2d-slice-times-vector
 *
 * a is a 2d-slice (M x N) i.e. (nn x na_m) where every ROW is CONTIGUOUSLY stored in memory.
 *
 * M number of rows    i.e. the length of a's column (nn)
 * N number of columns i.e. the length of a's row    (na_m)
 *
 * lda is the stride (na_m), the number of entries between two elements neighbouring elements of a column
 *
 *
 * b is a vector or fiber and should point to a contiguously stored memory region of length nn
 *
 * c is a vector or fiber and should point to a contiguously stored memory region of length na_m
 *
 * \note performs this with blas library
 *
*/
template<class value_t, class size_t>
static inline void gemv_row(
		value_t const*const __restrict a,
		value_t const*const __restrict b,
		value_t      *const __restrict c,
		size_t const M, // nn
		size_t const N, // na_m
		size_t const lda) // na_m usually as
{
	for(auto i = 0ul; i < M; ++i){ // over row
		auto const*const __restrict arow = a+i*lda;
		auto sum = value_t{};
		#pragma omp simd reduction (+:sum) // aligned (arow,b : 32)
		for(auto k = 0ul; k < N; ++k){ // over column
			sum += arow[k] * b[k];
		}
		c[i] = sum;
	}
}	

template<class value_t, class size_t>
static inline void gemv_row_parallel(
		value_t const*const __restrict a,
		value_t const*const __restrict b,
		value_t      *const __restrict c,
		size_t const M, // nn
		size_t const N, // na_m
		size_t const lda) // na_m usually as
{
	#pragma omp parallel for firstprivate(a,b,c,N,lda,M)
	for(auto i = 0ul; i < M; ++i){ // over row
		auto const*const __restrict arow = a+i*lda;
		auto sum = value_t{};
		#pragma omp simd reduction (+:sum) // aligned (arow,b : 32)
		for(auto k = 0ul; k < N; ++k){ // over column
			sum += arow[k] * b[k];
		}
		c[i] = sum;
	}
}	
/** \brief computes 2d-slice-times-vector
 *
 * a is a 2d-slice (M x N) i.e. (na_pia_1 x na_m or nn x na_m) where every COLUMN is CONTIGUOUSLY stored in memory.
 *
 * M number of rows    i.e. the length of a's column (na_pia_1 or nn)
 * N number of columns i.e. the length of a's row    (na_m)
 *
 * lda is the stride (wa_m), the number of entries between two elements neighbouring elements of a row
 *
 * b is a vector or fiber and should point to a contiguously stored memory region of length na_m
 *
 * c is a vector or fiber and should point to a contiguously stored memory region of length na_pia_1 or nn
 *
*/
template<class value_t, class size_t>
inline void gemv_col(
		value_t const*const __restrict a,
		value_t const*const __restrict b,
		value_t      *const __restrict c,
		size_t const M, // nn
		size_t const N, // na_m
		size_t const lda) // wa_m
{

	for(unsigned i = 0; i < N; ++i){
		auto const*const __restrict a0 = a+i*lda;
		auto      *const __restrict c0 = c;
		const auto bb = b[i];

		#pragma omp simd // aligned (c0,a0 : 32)
		for(unsigned j = 0; j < M; ++j){
			c0[j] += a0[j]  * bb;
		}
	}
}	



template<class value_t, class size_t>
void gemv_col_parallel(
		value_t const*const __restrict a,
		value_t const*const __restrict b,
		value_t      *const __restrict c,
		size_t const M,
		size_t const N,
		size_t const lda)
{
	constexpr auto MB = 256;
	const unsigned m = M/MB;
	const unsigned MBmod = M%MB;

	#pragma omp parallel  firstprivate(a,b,c, MB, m, MBmod, N, lda, M)
	{
		#pragma omp for schedule(dynamic)
		for(unsigned k = 0; k < m; ++k){
			auto const*const __restrict ak = a+k*MB;
			auto      *const __restrict ck = c+k*MB;

			for(unsigned i = 0; i < N; ++i){
				auto const*const __restrict ai = ak+i*lda;
				auto      *const __restrict ci = ck;
				const auto bb = b[i];

				#pragma omp simd safelen(MB)
				// aligned (ci,ai : 32)
				for(unsigned j = 0; j < MB; ++j){
					ci[j] += ai[j]  * bb;
				}
			}
		}

		#pragma omp single
		for(unsigned i = 0; i < N; ++i){
			auto const*const __restrict ai = a+i*lda+m*MB;
			auto      *const __restrict ci = c+m*MB;
			const auto bb = b[i];

			#pragma omp simd // aligned (ci,ai : 32)
			for(unsigned j = 0; j < MBmod; ++j){
				ci[j] += ai[j]  * bb;
			}
		}

	}
}

/** \brief computes 2d-slice-times-vector
 *
 * the same as above only using basic linear algebra subroutines
 *
 * \note performs this with blas library
 *
*/
template<class value_t, class size_t>
inline void gemv_col_blas(
		value_t const*const __restrict a,
		value_t const*const __restrict b,
		value_t      *const __restrict c,
		size_t const M,
		size_t const N,
		size_t const lda)
{
				// CblasColMajor CblasNoTrans      m         n     alpha  a   lda   x  incx  beta  y   incy
#if defined USE_INTELBLAS || defined USE_OPENBLAS
	if constexpr      ( std::is_same<value_t,float>::value )
		cblas_sgemv(CblasColMajor, CblasNoTrans, M,  N, 1.0f,  const_cast<float*const>(a),  lda, const_cast<float*const> (b), 1,  0.0f, const_cast<float*const> (c),  1);
	else if constexpr ( std::is_same<value_t,double>::value )
		cblas_dgemv(CblasColMajor, CblasNoTrans, M,  N, 1.0 ,  const_cast<double*const>(a),  lda, const_cast<double*const>(b), 1,  0.0 , const_cast<double*const>(c),  1);
	else
		gemv_col(a,b,c,M,N,lda);
#else
	gemv_col(a,b,c,M,N,lda);
#endif
}


/** \brief computes 2d-slice-times-vector
 *
 * the same as above only using basic linear algebra subroutines
 *
 * \note performs this with blas library
 *
*/
template<class value_t, class size_t>
inline void gemv_row_blas(
		value_t const*const __restrict a,
		value_t const*const __restrict b,
		value_t      *const __restrict c,
		size_t const M, // nn
		size_t const N, // na_m
		size_t const lda) // na_m usually as
{
		// CblasRowMajor CblasNoTrans      m         n     alpha  a   lda   x  incx  beta  y   incy
#if defined USE_INTELBLAS || defined USE_OPENBLAS
	if constexpr      ( std::is_same<value_t,float>::value )
		cblas_sgemv(CblasRowMajor, CblasNoTrans, M,  N, 1.0f,  const_cast<float*const>(a),  lda, const_cast<float*const> (b), 1,  0.0f, const_cast<float*const> (c),  1);
	else if constexpr ( std::is_same<value_t,double>::value )
		cblas_dgemv(CblasRowMajor, CblasNoTrans, M,  N, 1.0 ,  const_cast<double*const>(a),  lda, const_cast<double*const>(b), 1,  0.0 , const_cast<double*const>(c),  1);
	else
		gemv_row(a,b,c,M,N,lda);
#else
	gemv_row(a,b,c,M,N,lda);
#endif
}

template<class value_t, class size_t>
inline void dot(
	value_t const*const __restrict a,
	value_t const*const __restrict b,
	value_t      *const __restrict c,
	size_t const M) // nn
{
	auto sum = value_t{};
	#pragma omp simd reduction (+:sum)
	for(auto k = 0ul; k < M; ++k){
		sum += a[k] * b[k];
	}
	c[0] = sum;
}


template<class value_t, class size_t>
inline void dot_parallel(
	value_t const*const __restrict a,
	value_t const*const __restrict b,
	value_t      *const __restrict c,
	size_t const M) // nn
{
	auto sum = value_t{};
	#pragma omp parallel for firstprivate(a,b) reduction (+:sum)
	for(auto k = 0ul; k < M; ++k){
		sum += a[k] * b[k];
	}
	c[0] = sum;
}


template<class size_t>
inline auto compute_nfull(size_t const*const na, size_t p)
{
	return std::accumulate( na, na+p, 1ul, std::multiplies<>()  );
}


// value_t         value type of the elements
// execution_t  std::tuple of optimization types
//template<class value_t, class execution_t>
//struct MatrixTimesVector;
//template<class value_t>
//struct MatrixTimesVector<value_t,sequential_tag>


template<class value_t, class size_t, class execution_policy>
inline void mtv(
			execution_policy,
			size_t const m, size_t const p,
			value_t const*const a, size_t const*const na,     size_t const*const /*wa*/, size_t const*const pia,
			value_t const*const b, size_t const*const /*nb*/,
			value_t      *const c, size_t const*const /*nc*/, size_t const*const /*wc*/, size_t const*const /*pic*/
			);


//template<class value_t, size_t>
//struct MatrixTimesVector<value_t,size_t, sequential_tag>

template<class value_t, class size_t>
inline void mtv(
			execution::sequential_policy,
			size_t const m, size_t const p,
			value_t const*const a, size_t const*const na,     size_t const*const /*wa*/, size_t const*const pia,
			value_t const*const b, size_t const*const /*nb*/,
			value_t      *const c, size_t const*const /*nc*/, size_t const*const /*wc*/, size_t const*const /*pic*/
			)
{
	auto n = compute_nfull(na,p) / na[m-1];
	
	     if(is_case<1>(p,m,pia)) dot     (a,b,c,na[0]);
	else if(is_case<2>(p,m,pia)) gemv_row(a,b,c,na[1],na[0],na[0] ); // first-order (column-major)
	else if(is_case<3>(p,m,pia)) gemv_col(a,b,c,na[0],na[1],na[0] ); // first-order (column-major)
	else if(is_case<4>(p,m,pia)) gemv_col(a,b,c,na[1],na[0],na[1] ); // last-order  (row-major)
	else if(is_case<5>(p,m,pia)) gemv_row(a,b,c,na[0],na[1],na[1] ); // last-order  (row-major)
	else if(is_case<6>(p,m,pia)) gemv_row(a,b,c,n,na[m-1],na[m-1]);
	else if(is_case<7>(p,m,pia)) gemv_col(a,b,c,n,na[m-1],n);

}



//template<class value_t>
//struct MatrixTimesVector<value_t,parallel_tag>
template<class value_t, class size_t>
inline void mtv(
			execution::parallel_policy,
			size_t const m, size_t const p,
			value_t const*const a, size_t const*const na,     size_t const*const /*wa*/, size_t const*const pia,
			value_t const*const b, size_t const*const /*nb*/,
			value_t      *const c, size_t const*const /*nc*/, size_t const*const /*wc*/, size_t const*const /*pic*/
			)
{
	
	auto n = compute_nfull(na,p) / na[m-1];
	
	     if(is_case<1>(p,m,pia)) dot_parallel     (a,b,c,na[0]);
	else if(is_case<2>(p,m,pia)) gemv_row         (a,b,c,na[1],na[0],na[0] ); // first-order (column-major)
	else if(is_case<3>(p,m,pia)) gemv_col_parallel(a,b,c,na[0],na[1],na[0] ); // first-order (column-major)
	else if(is_case<4>(p,m,pia)) gemv_col_parallel(a,b,c,na[1],na[0],na[1] ); // last-order  (row-major)
	else if(is_case<5>(p,m,pia)) gemv_row         (a,b,c,na[0],na[1],na[1] ); // last-order  (row-major)
	else if(is_case<6>(p,m,pia)) gemv_row         (a,b,c,n,na[m-1],na[m-1]);
	else if(is_case<7>(p,m,pia)) gemv_col_parallel(a,b,c,n,na[m-1],n);		
	
}




//template<class value_t>
//struct MatrixTimesVector<value_t,blas_tag>
template<class value_t, class size_t>
inline void mtv(
			execution::parallel_blas_policy,
			size_t const m, size_t const p,
			value_t const*const a, size_t const*const na,     size_t const*const /*wa*/, size_t const*const pia,
			value_t const*const b, size_t const*const /*nb*/,
			value_t      *const c, size_t const*const /*nc*/, size_t const*const /*wc*/, size_t const*const /*pic*/
			)
{

	auto n = compute_nfull(na,p) / na[m-1];
	
	     if(is_case<1>(p,m,pia)) dot_parallel  (a,b,c,na[0]);
	else if(is_case<2>(p,m,pia)) gemv_row_blas (a,b,c,na[1],na[0],na[0] ); // first-order (column-major)
	else if(is_case<3>(p,m,pia)) gemv_col_blas (a,b,c,na[0],na[1],na[0] ); // first-order (column-major)
	else if(is_case<4>(p,m,pia)) gemv_col_blas (a,b,c,na[1],na[0],na[1] ); // last-order  (row-major)
	else if(is_case<5>(p,m,pia)) gemv_row_blas (a,b,c,na[0],na[1],na[1] ); // last-order  (row-major)
	else if(is_case<6>(p,m,pia)) gemv_row_blas (a,b,c,n,na[m-1],na[m-1]);
	else if(is_case<7>(p,m,pia)) gemv_col_blas (a,b,c,n,na[m-1],n);	
	
}


template<typename Kernel, typename Partition, 
	typename SizeType, typename ValueType
>
inline void block_mtv_col(ValueType* c, SizeType const* nc, SizeType const* wc,
	ValueType const* a, SizeType const* na, SizeType const* wa,
	ValueType const* b, SizeType const* nb, SizeType const* wb,
	Kernel ker, Partition par
) noexcept
{
	auto ai = a;
	auto bi = b;
	auto ci = c;

	auto const m = na[0];
	auto const k = na[1];
	
	par(m,k,1);

	auto const BM = par.M();
	auto const BK = par.K();
	#pragma omp parallel for schedule(dynamic)
	for( auto i = 0ul; i < m; i += BM ){
		auto const ib = std::min( m - i, BM );

		auto ak = ai + i * wa[0];
		auto bk = bi;
		auto ck = ci + i * wc[0];

		for( auto j = 0ul; j < k; j += BK ){
			auto const jb = std::min( k - j, BK );
			
			SizeType const nta[] = {ib, jb};
			SizeType const ntb[] = {jb, 1};
			SizeType const ntc[] = {ib, 1};

			ker(
				ck, ntc, wc,
				ak, nta, wa,
				bk, ntb, wb
			);

			ak += wa[1] * jb;
			bk += jb;
		}

	}

}

template<typename Kernel, typename Partition, 
	typename SizeType, typename ValueType
>
inline void block_mtv_row(ValueType* c, SizeType const* nc, SizeType const* wc,
	ValueType const* a, SizeType const* na, SizeType const* wa,
	ValueType const* b, SizeType const* nb, SizeType const* wb,
	Kernel ker, Partition par
) noexcept
{
	auto ai = a;
	auto bi = b;
	auto ci = c;

	auto const m = na[1];
	auto const k = na[0];

	par(m,k,1);
	auto const BM = par.M();
	auto const BK = par.K();

	SizeType const wta[] = {1, std::max(wa[0],wa[1])};
	SizeType const wtb[] = {1, std::max(wb[0],wb[1])};
	SizeType const wtc[] = {1, std::max(wc[0],wc[1])};

	#pragma omp parallel for schedule(dynamic)
	for( auto i = 0ul; i < k; i += BK ){
		auto ib = std::min(k - i, BK);

		auto ak = ai + i * wta[1];
		auto bk = bi;
		auto ck = ci + i;
		for( auto j = 0ul; j < m; j += BM ){
			auto jb = std::min( m - j, BM );

			SizeType const nta[] = {jb, ib};
			SizeType const ntb[] = {jb, 1};
			SizeType const ntc[] = {ib, 1};

			ker(
				ck, ntc, wc,
				ak, nta, wa,
				bk, ntb, wb
			);
			ak += wta[0] * jb;
			bk += jb;
		}
	}

}


} // namespace tlib::detail

#endif // TLIB_DETAIL_MV_H
