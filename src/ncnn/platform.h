#pragma once

#if NCNN_FORCE_INLINE
#ifdef _MSC_VER
#define NCNN_FORCEINLINE __forceinline
#elif defined(__GNUC__)
#define NCNN_FORCEINLINE inline __attribute__((__always_inline__))
#elif defined(__CLANG__)
#if __has_attribute(__always_inline__)
#define NCNN_FORCEINLINE inline __attribute__((__always_inline__))
#else
#define NCNN_FORCEINLINE inline
#endif
#else
#define NCNN_FORCEINLINE inline
#endif
#else
#define NCNN_FORCEINLINE inline
#endif

#include "ncnn/ncnn_export.h"