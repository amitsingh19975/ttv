#ifndef TLIB_DETAIL_KERNEL_X86_H
#define TLIB_DETAIL_KERNEL_X86_H

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

#include "x86_col.h"
#include "x86_row.h"

#endif // TLIB_DETAIL_KERNEL_X86_H
