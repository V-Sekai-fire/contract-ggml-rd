#ifndef SLANG_CPP_PRELUDE_H
#define SLANG_CPP_PRELUDE_H

// Because the signature of isnan, isfinite, and is isinf changed in C++, we use the macro
// to use the version in the std namespace.
// https://stackoverflow.com/questions/39130040/cmath-hides-isnan-in-math-h-in-c14-c11

#ifdef SLANG_LLVM
#ifndef SLANG_LLVM_H
#define SLANG_LLVM_H

// TODO(JS):
// Disable exception declspecs, as not supported on LLVM without some extra options.
// We could enable with `-fms-extensions`
#define SLANG_DISABLE_EXCEPTIONS 1

#ifndef SLANG_PRELUDE_ASSERT
#ifdef SLANG_PRELUDE_ENABLE_ASSERT
extern "C" void assertFailure(const char* msg);
#define SLANG_PRELUDE_EXPECT(VALUE, MSG) \
    if (VALUE)                           \
    {                                    \
    }                                    \
    else                                 \
        assertFailure("assertion failed: '" MSG "'")
#define SLANG_PRELUDE_ASSERT(VALUE) SLANG_PRELUDE_EXPECT(VALUE, #VALUE)
#else // SLANG_PRELUDE_ENABLE_ASSERT
#define SLANG_PRELUDE_EXPECT(VALUE, MSG)
#define SLANG_PRELUDE_ASSERT(x)
#endif // SLANG_PRELUDE_ENABLE_ASSERT
#endif

/*
Taken from stddef.h
*/

typedef __PTRDIFF_TYPE__ ptrdiff_t;
typedef __SIZE_TYPE__ size_t;
typedef __SIZE_TYPE__ rsize_t;

// typedef __WCHAR_TYPE__ wchar_t;

#if defined(__need_NULL)
#undef NULL
#ifdef __cplusplus
#if !defined(__MINGW32__) && !defined(_MSC_VER)
#define NULL __null
#else
#define NULL 0
#endif
#else
#define NULL ((void*)0)
#endif
#ifdef __cplusplus
#if defined(_MSC_EXTENSIONS) && defined(_NATIVE_NULLPTR_SUPPORTED)
namespace std
{
typedef decltype(nullptr) nullptr_t;
}
using ::std::nullptr_t;
#endif
#endif
#undef __need_NULL
#endif /* defined(__need_NULL) */


/*
The following are taken verbatim from stdint.h from Clang in LLVM. Only 8/16/32/64 types are needed.
*/

// LLVM/Clang types such that we can use LLVM/Clang without headers for C++ output from Slang

#ifdef __INT64_TYPE__
#ifndef __int8_t_defined /* glibc sys/types.h also defines int64_t*/
typedef __INT64_TYPE__ int64_t;
#endif /* __int8_t_defined */
typedef __UINT64_TYPE__ uint64_t;
#define __int_least64_t int64_t
#define __uint_least64_t uint64_t
#endif /* __INT64_TYPE__ */

#ifdef __int_least64_t
typedef __int_least64_t int_least64_t;
typedef __uint_least64_t uint_least64_t;
typedef __int_least64_t int_fast64_t;
typedef __uint_least64_t uint_fast64_t;
#endif /* __int_least64_t */

#ifdef __INT32_TYPE__

#ifndef __int8_t_defined /* glibc sys/types.h also defines int32_t*/
typedef __INT32_TYPE__ int32_t;
#endif /* __int8_t_defined */

#ifndef __uint32_t_defined /* more glibc compatibility */
#define __uint32_t_defined
typedef __UINT32_TYPE__ uint32_t;
#endif /* __uint32_t_defined */

#define __int_least32_t int32_t
#define __uint_least32_t uint32_t
#endif /* __INT32_TYPE__ */

#ifdef __int_least32_t
typedef __int_least32_t int_least32_t;
typedef __uint_least32_t uint_least32_t;
typedef __int_least32_t int_fast32_t;
typedef __uint_least32_t uint_fast32_t;
#endif /* __int_least32_t */

#ifdef __INT16_TYPE__
#ifndef __int8_t_defined /* glibc sys/types.h also defines int16_t*/
typedef __INT16_TYPE__ int16_t;
#endif /* __int8_t_defined */
typedef __UINT16_TYPE__ uint16_t;
#define __int_least16_t int16_t
#define __uint_least16_t uint16_t
#endif /* __INT16_TYPE__ */

#ifdef __int_least16_t
typedef __int_least16_t int_least16_t;
typedef __uint_least16_t uint_least16_t;
typedef __int_least16_t int_fast16_t;
typedef __uint_least16_t uint_fast16_t;
#endif /* __int_least16_t */

#ifdef __INT8_TYPE__
#ifndef __int8_t_defined /* glibc sys/types.h also defines int8_t*/
typedef __INT8_TYPE__ int8_t;
#endif /* __int8_t_defined */
typedef __UINT8_TYPE__ uint8_t;
#define __int_least8_t int8_t
#define __uint_least8_t uint8_t
#endif /* __INT8_TYPE__ */

#ifdef __int_least8_t
typedef __int_least8_t int_least8_t;
typedef __uint_least8_t uint_least8_t;
typedef __int_least8_t int_fast8_t;
typedef __uint_least8_t uint_fast8_t;
#endif /* __int_least8_t */

/* prevent glibc sys/types.h from defining conflicting types */
#ifndef __int8_t_defined
#define __int8_t_defined
#endif /* __int8_t_defined */

/* C99 7.18.1.4 Integer types capable of holding object pointers.
 */
#define __stdint_join3(a, b, c) a##b##c

#ifndef _INTPTR_T
#ifndef __intptr_t_defined
typedef __INTPTR_TYPE__ intptr_t;
#define __intptr_t_defined
#define _INTPTR_T
#endif
#endif

#ifndef _UINTPTR_T
typedef __UINTPTR_TYPE__ uintptr_t;
#define _UINTPTR_T
#endif

/* C99 7.18.1.5 Greatest-width integer types.
 */
typedef __INTMAX_TYPE__ intmax_t;
typedef __UINTMAX_TYPE__ uintmax_t;

/* C99 7.18.4 Macros for minimum-width integer constants.
 *
 * The standard requires that integer constant macros be defined for all the
 * minimum-width types defined above. As 8-, 16-, 32-, and 64-bit minimum-width
 * types are required, the corresponding integer constant macros are defined
 * here. This implementation also defines minimum-width types for every other
 * integer width that the target implements, so corresponding macros are
 * defined below, too.
 *
 * These macros are defined using the same successive-shrinking approach as
 * the type definitions above. It is likewise important that macros are defined
 * in order of decending width.
 *
 * Note that C++ should not check __STDC_CONSTANT_MACROS here, contrary to the
 * claims of the C standard (see C++ 18.3.1p2, [cstdint.syn]).
 */

#define __int_c_join(a, b) a##b
#define __int_c(v, suffix) __int_c_join(v, suffix)
#define __uint_c(v, suffix) __int_c_join(v##U, suffix)

#ifdef __INT64_TYPE__
#ifdef __INT64_C_SUFFIX__
#define __int64_c_suffix __INT64_C_SUFFIX__
#else
#undef __int64_c_suffix
#endif /* __INT64_C_SUFFIX__ */
#endif /* __INT64_TYPE__ */

#ifdef __int_least64_t
#ifdef __int64_c_suffix
#define INT64_C(v) __int_c(v, __int64_c_suffix)
#define UINT64_C(v) __uint_c(v, __int64_c_suffix)
#else
#define INT64_C(v) v
#define UINT64_C(v) v##U
#endif /* __int64_c_suffix */
#endif /* __int_least64_t */


#ifdef __INT32_TYPE__
#ifdef __INT32_C_SUFFIX__
#define __int32_c_suffix __INT32_C_SUFFIX__
#else
#undef __int32_c_suffix
#endif /* __INT32_C_SUFFIX__ */
#endif /* __INT32_TYPE__ */

#ifdef __int_least32_t
#ifdef __int32_c_suffix
#define INT32_C(v) __int_c(v, __int32_c_suffix)
#define UINT32_C(v) __uint_c(v, __int32_c_suffix)
#else
#define INT32_C(v) v
#define UINT32_C(v) v##U
#endif /* __int32_c_suffix */
#endif /* __int_least32_t */

#ifdef __INT16_TYPE__
#ifdef __INT16_C_SUFFIX__
#define __int16_c_suffix __INT16_C_SUFFIX__
#else
#undef __int16_c_suffix
#endif /* __INT16_C_SUFFIX__ */
#endif /* __INT16_TYPE__ */

#ifdef __int_least16_t
#ifdef __int16_c_suffix
#define INT16_C(v) __int_c(v, __int16_c_suffix)
#define UINT16_C(v) __uint_c(v, __int16_c_suffix)
#else
#define INT16_C(v) v
#define UINT16_C(v) v##U
#endif /* __int16_c_suffix */
#endif /* __int_least16_t */


#ifdef __INT8_TYPE__
#ifdef __INT8_C_SUFFIX__
#define __int8_c_suffix __INT8_C_SUFFIX__
#else
#undef __int8_c_suffix
#endif /* __INT8_C_SUFFIX__ */
#endif /* __INT8_TYPE__ */

#ifdef __int_least8_t
#ifdef __int8_c_suffix
#define INT8_C(v) __int_c(v, __int8_c_suffix)
#define UINT8_C(v) __uint_c(v, __int8_c_suffix)
#else
#define INT8_C(v) v
#define UINT8_C(v) v##U
#endif /* __int8_c_suffix */
#endif /* __int_least8_t */

/* C99 7.18.2.1 Limits of exact-width integer types.
 * C99 7.18.2.2 Limits of minimum-width integer types.
 * C99 7.18.2.3 Limits of fastest minimum-width integer types.
 *
 * The presence of limit macros are completely optional in C99.  This
 * implementation defines limits for all of the types (exact- and
 * minimum-width) that it defines above, using the limits of the minimum-width
 * type for any types that do not have exact-width representations.
 *
 * As in the type definitions, this section takes an approach of
 * successive-shrinking to determine which limits to use for the standard (8,
 * 16, 32, 64) bit widths when they don't have exact representations. It is
 * therefore important that the definitions be kept in order of decending
 * widths.
 *
 * Note that C++ should not check __STDC_LIMIT_MACROS here, contrary to the
 * claims of the C standard (see C++ 18.3.1p2, [cstdint.syn]).
 */

#ifdef __INT64_TYPE__
#define INT64_MAX INT64_C(9223372036854775807)
#define INT64_MIN (-INT64_C(9223372036854775807) - 1)
#define UINT64_MAX UINT64_C(18446744073709551615)
#define __INT_LEAST64_MIN INT64_MIN
#define __INT_LEAST64_MAX INT64_MAX
#define __UINT_LEAST64_MAX UINT64_MAX
#endif /* __INT64_TYPE__ */

#ifdef __INT_LEAST64_MIN
#define INT_LEAST64_MIN __INT_LEAST64_MIN
#define INT_LEAST64_MAX __INT_LEAST64_MAX
#define UINT_LEAST64_MAX __UINT_LEAST64_MAX
#define INT_FAST64_MIN __INT_LEAST64_MIN
#define INT_FAST64_MAX __INT_LEAST64_MAX
#define UINT_FAST64_MAX __UINT_LEAST64_MAX
#endif /* __INT_LEAST64_MIN */

#ifdef __INT32_TYPE__
#define INT32_MAX INT32_C(2147483647)
#define INT32_MIN (-INT32_C(2147483647) - 1)
#define UINT32_MAX UINT32_C(4294967295)
#define __INT_LEAST32_MIN INT32_MIN
#define __INT_LEAST32_MAX INT32_MAX
#define __UINT_LEAST32_MAX UINT32_MAX
#endif /* __INT32_TYPE__ */

#ifdef __INT_LEAST32_MIN
#define INT_LEAST32_MIN __INT_LEAST32_MIN
#define INT_LEAST32_MAX __INT_LEAST32_MAX
#define UINT_LEAST32_MAX __UINT_LEAST32_MAX
#define INT_FAST32_MIN __INT_LEAST32_MIN
#define INT_FAST32_MAX __INT_LEAST32_MAX
#define UINT_FAST32_MAX __UINT_LEAST32_MAX
#endif /* __INT_LEAST32_MIN */

#ifdef __INT16_TYPE__
#define INT16_MAX INT16_C(32767)
#define INT16_MIN (-INT16_C(32767) - 1)
#define UINT16_MAX UINT16_C(65535)
#define __INT_LEAST16_MIN INT16_MIN
#define __INT_LEAST16_MAX INT16_MAX
#define __UINT_LEAST16_MAX UINT16_MAX
#endif /* __INT16_TYPE__ */

#ifdef __INT_LEAST16_MIN
#define INT_LEAST16_MIN __INT_LEAST16_MIN
#define INT_LEAST16_MAX __INT_LEAST16_MAX
#define UINT_LEAST16_MAX __UINT_LEAST16_MAX
#define INT_FAST16_MIN __INT_LEAST16_MIN
#define INT_FAST16_MAX __INT_LEAST16_MAX
#define UINT_FAST16_MAX __UINT_LEAST16_MAX
#endif /* __INT_LEAST16_MIN */


#ifdef __INT8_TYPE__
#define INT8_MAX INT8_C(127)
#define INT8_MIN (-INT8_C(127) - 1)
#define UINT8_MAX UINT8_C(255)
#define __INT_LEAST8_MIN INT8_MIN
#define __INT_LEAST8_MAX INT8_MAX
#define __UINT_LEAST8_MAX UINT8_MAX
#endif /* __INT8_TYPE__ */

#ifdef __INT_LEAST8_MIN
#define INT_LEAST8_MIN __INT_LEAST8_MIN
#define INT_LEAST8_MAX __INT_LEAST8_MAX
#define UINT_LEAST8_MAX __UINT_LEAST8_MAX
#define INT_FAST8_MIN __INT_LEAST8_MIN
#define INT_FAST8_MAX __INT_LEAST8_MAX
#define UINT_FAST8_MAX __UINT_LEAST8_MAX
#endif /* __INT_LEAST8_MIN */

/* Some utility macros */
#define __INTN_MIN(n) __stdint_join3(INT, n, _MIN)
#define __INTN_MAX(n) __stdint_join3(INT, n, _MAX)
#define __UINTN_MAX(n) __stdint_join3(UINT, n, _MAX)
#define __INTN_C(n, v) __stdint_join3(INT, n, _C(v))
#define __UINTN_C(n, v) __stdint_join3(UINT, n, _C(v))

/* C99 7.18.2.4 Limits of integer types capable of holding object pointers. */
/* C99 7.18.3 Limits of other integer types. */

#define INTPTR_MIN (-__INTPTR_MAX__ - 1)
#define INTPTR_MAX __INTPTR_MAX__
#define UINTPTR_MAX __UINTPTR_MAX__
#define PTRDIFF_MIN (-__PTRDIFF_MAX__ - 1)
#define PTRDIFF_MAX __PTRDIFF_MAX__
#define SIZE_MAX __SIZE_MAX__

/* ISO9899:2011 7.20 (C11 Annex K): Define RSIZE_MAX if __STDC_WANT_LIB_EXT1__
 * is enabled. */
#if defined(__STDC_WANT_LIB_EXT1__) && __STDC_WANT_LIB_EXT1__ >= 1
#define RSIZE_MAX (SIZE_MAX >> 1)
#endif

/* C99 7.18.2.5 Limits of greatest-width integer types. */
#define INTMAX_MIN (-__INTMAX_MAX__ - 1)
#define INTMAX_MAX __INTMAX_MAX__
#define UINTMAX_MAX __UINTMAX_MAX__

/* C99 7.18.3 Limits of other integer types. */
#define SIG_ATOMIC_MIN __INTN_MIN(__SIG_ATOMIC_WIDTH__)
#define SIG_ATOMIC_MAX __INTN_MAX(__SIG_ATOMIC_WIDTH__)
#ifdef __WINT_UNSIGNED__
#define WINT_MIN __UINTN_C(__WINT_WIDTH__, 0)
#define WINT_MAX __UINTN_MAX(__WINT_WIDTH__)
#else
#define WINT_MIN __INTN_MIN(__WINT_WIDTH__)
#define WINT_MAX __INTN_MAX(__WINT_WIDTH__)
#endif

#ifndef WCHAR_MAX
#define WCHAR_MAX __WCHAR_MAX__
#endif
#ifndef WCHAR_MIN
#if __WCHAR_MAX__ == __INTN_MAX(__WCHAR_WIDTH__)
#define WCHAR_MIN __INTN_MIN(__WCHAR_WIDTH__)
#else
#define WCHAR_MIN __UINTN_C(__WCHAR_WIDTH__, 0)
#endif
#endif

/* 7.18.4.2 Macros for greatest-width integer constants. */
#define INTMAX_C(v) __int_c(v, __INTMAX_C_SUFFIX__)
#define UINTMAX_C(v) __int_c(v, __UINTMAX_C_SUFFIX__)


#endif // SLANG_LLVM_H

#else // SLANG_LLVM
#if SLANG_GCC_FAMILY && __GNUC__ < 6
#include <cmath>
#define SLANG_PRELUDE_STD std::
#else
#include <math.h>
#define SLANG_PRELUDE_STD
#endif

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#endif // SLANG_LLVM

// Is intptr_t not equal to equal-width sized integer type?
#if defined(__APPLE__)
#define SLANG_INTPTR_TYPE_IS_DISTINCT 1
#else
#define SLANG_INTPTR_TYPE_IS_DISTINCT 0
#endif

#if defined(_MSC_VER)
#define SLANG_PRELUDE_SHARED_LIB_EXPORT __declspec(dllexport)
#else
#define SLANG_PRELUDE_SHARED_LIB_EXPORT __attribute__((__visibility__("default")))
// #   define SLANG_PRELUDE_SHARED_LIB_EXPORT __attribute__ ((dllexport))
// __attribute__((__visibility__("default")))
#endif

#ifdef __cplusplus
#define SLANG_PRELUDE_EXTERN_C extern "C"
#define SLANG_PRELUDE_EXTERN_C_START \
    extern "C"                       \
    {
#define SLANG_PRELUDE_EXTERN_C_END }
#else
#define SLANG_PRELUDE_EXTERN_C
#define SLANG_PRELUDE_EXTERN_C_START
#define SLANG_PRELUDE_EXTERN_C_END
#endif

#define SLANG_PRELUDE_EXPORT SLANG_PRELUDE_EXTERN_C SLANG_PRELUDE_SHARED_LIB_EXPORT
#define SLANG_PRELUDE_EXPORT_START SLANG_PRELUDE_EXTERN_C_START SLANG_PRELUDE_SHARED_LIB_EXPORT
#define SLANG_PRELUDE_EXPORT_END SLANG_PRELUDE_EXTERN_C_END

#ifndef INFINITY
// Must overflow for double
#define INFINITY float(1e+300 * 1e+300)
#endif

#ifndef SLANG_INFINITY
#define SLANG_INFINITY INFINITY
#endif

// Detect the compiler type

#ifndef SLANG_COMPILER
#define SLANG_COMPILER

/*
Compiler defines, see http://sourceforge.net/p/predef/wiki/Compilers/
NOTE that SLANG_VC holds the compiler version - not just 1 or 0
*/
#if defined(_MSC_VER)
#if _MSC_VER >= 1900
#define SLANG_VC 14
#elif _MSC_VER >= 1800
#define SLANG_VC 12
#elif _MSC_VER >= 1700
#define SLANG_VC 11
#elif _MSC_VER >= 1600
#define SLANG_VC 10
#elif _MSC_VER >= 1500
#define SLANG_VC 9
#else
#error "unknown version of Visual C++ compiler"
#endif
#elif defined(__clang__)
#define SLANG_CLANG 1
#elif defined(__SNC__)
#define SLANG_SNC 1
#elif defined(__ghs__)
#define SLANG_GHS 1
#elif defined(__GNUC__) /* note: __clang__, __SNC__, or __ghs__ imply __GNUC__ */
#define SLANG_GCC 1
#else
#error "unknown compiler"
#endif
/*
Any compilers not detected by the above logic are now now explicitly zeroed out.
*/
#ifndef SLANG_VC
#define SLANG_VC 0
#endif
#ifndef SLANG_CLANG
#define SLANG_CLANG 0
#endif
#ifndef SLANG_SNC
#define SLANG_SNC 0
#endif
#ifndef SLANG_GHS
#define SLANG_GHS 0
#endif
#ifndef SLANG_GCC
#define SLANG_GCC 0
#endif
#endif /* SLANG_COMPILER */

/*
The following section attempts to detect the target platform being compiled for.

If an application defines `SLANG_PLATFORM` before including this header,
they take responsibility for setting any compiler-dependent macros
used later in the file.

Most applications should not need to touch this section.
*/
#ifndef SLANG_PLATFORM
#define SLANG_PLATFORM
/**
Operating system defines, see http://sourceforge.net/p/predef/wiki/OperatingSystems/
*/
#if defined(WINAPI_FAMILY) && WINAPI_FAMILY == WINAPI_PARTITION_APP
#define SLANG_WINRT 1 /* Windows Runtime, either on Windows RT or Windows 8 */
#elif defined(XBOXONE)
#define SLANG_XBOXONE 1
#elif defined(_WIN64) /* note: XBOXONE implies _WIN64 */
#define SLANG_WIN64 1
#elif defined(_M_PPC)
#define SLANG_X360 1
#elif defined(_WIN32) /* note: _M_PPC implies _WIN32 */
#define SLANG_WIN32 1
#elif defined(__ANDROID__)
#define SLANG_ANDROID 1
#elif defined(__linux__) || defined(__CYGWIN__) /* note: __ANDROID__ implies __linux__ */
#define SLANG_LINUX 1
#elif defined(__APPLE__) && !defined(SLANG_LLVM)
#include "TargetConditionals.h"
#if TARGET_OS_MAC
#define SLANG_OSX 1
#else
#define SLANG_IOS 1
#endif
#elif defined(__APPLE__)
// On `slang-llvm` we can't inclue "TargetConditionals.h" in general, so for now assume its
// OSX.
#define SLANG_OSX 1
#elif defined(__CELLOS_LV2__)
#define SLANG_PS3 1
#elif defined(__ORBIS__)
#define SLANG_PS4 1
#elif defined(__SNC__) && defined(__arm__)
#define SLANG_PSP2 1
#elif defined(__ghs__)
#define SLANG_WIIU 1
#else
#error "unknown target platform"
#endif


/*
Any platforms not detected by the above logic are now now explicitly zeroed out.
*/
#ifndef SLANG_WINRT
#define SLANG_WINRT 0
#endif
#ifndef SLANG_XBOXONE
#define SLANG_XBOXONE 0
#endif
#ifndef SLANG_WIN64
#define SLANG_WIN64 0
#endif
#ifndef SLANG_X360
#define SLANG_X360 0
#endif
#ifndef SLANG_WIN32
#define SLANG_WIN32 0
#endif
#ifndef SLANG_ANDROID
#define SLANG_ANDROID 0
#endif
#ifndef SLANG_LINUX
#define SLANG_LINUX 0
#endif
#ifndef SLANG_IOS
#define SLANG_IOS 0
#endif
#ifndef SLANG_OSX
#define SLANG_OSX 0
#endif
#ifndef SLANG_PS3
#define SLANG_PS3 0
#endif
#ifndef SLANG_PS4
#define SLANG_PS4 0
#endif
#ifndef SLANG_PSP2
#define SLANG_PSP2 0
#endif
#ifndef SLANG_WIIU
#define SLANG_WIIU 0
#endif
#endif /* SLANG_PLATFORM */

/* Shorthands for "families" of compilers/platforms */
#define SLANG_GCC_FAMILY (SLANG_CLANG || SLANG_SNC || SLANG_GHS || SLANG_GCC)
#define SLANG_WINDOWS_FAMILY (SLANG_WINRT || SLANG_WIN32 || SLANG_WIN64)
#define SLANG_MICROSOFT_FAMILY (SLANG_XBOXONE || SLANG_X360 || SLANG_WINDOWS_FAMILY)
#define SLANG_LINUX_FAMILY (SLANG_LINUX || SLANG_ANDROID)
#define SLANG_APPLE_FAMILY (SLANG_IOS || SLANG_OSX) /* equivalent to #if __APPLE__ */
#define SLANG_UNIX_FAMILY \
    (SLANG_LINUX_FAMILY || SLANG_APPLE_FAMILY) /* shortcut for unix/posix platforms */

// GCC Specific
#if SLANG_GCC_FAMILY

#if INTPTR_MAX == INT64_MAX
#define SLANG_64BIT 1
#else
#define SLANG_64BIT 0
#endif

#define SLANG_BREAKPOINT(id) __builtin_trap()

// Use this macro instead of offsetof, because gcc produces warning if offsetof is used on a
// non POD type, even though it produces the correct result
#define SLANG_OFFSET_OF(T, ELEMENT) (size_t(&((T*)1)->ELEMENT) - 1)
#endif // SLANG_GCC_FAMILY

// Microsoft VC specific
#if SLANG_VC

#define SLANG_BREAKPOINT(id) __debugbreak();

#endif // SLANG_VC

// Default impls

#ifndef SLANG_OFFSET_OF
#define SLANG_OFFSET_OF(X, Y) offsetof(X, Y)
#endif

#ifndef SLANG_BREAKPOINT
// Make it crash with a write to 0!
#define SLANG_BREAKPOINT(id) (*((int*)0) = int(id));
#endif

// If slang.h has been included we don't need any of these definitions
#ifndef SLANG_H

/* Macro for declaring if a method is no throw. Should be set before the return parameter. */
#ifndef SLANG_NO_THROW
#if SLANG_WINDOWS_FAMILY && !defined(SLANG_DISABLE_EXCEPTIONS)
#define SLANG_NO_THROW __declspec(nothrow)
#endif
#endif
#ifndef SLANG_NO_THROW
#define SLANG_NO_THROW
#endif

/* The `SLANG_STDCALL` and `SLANG_MCALL` defines are used to set the calling
convention for interface methods.
*/
#ifndef SLANG_STDCALL
#if SLANG_MICROSOFT_FAMILY
#define SLANG_STDCALL __stdcall
#else
#define SLANG_STDCALL
#endif
#endif
#ifndef SLANG_MCALL
#define SLANG_MCALL SLANG_STDCALL
#endif

#ifndef SLANG_FORCE_INLINE
#define SLANG_FORCE_INLINE inline
#endif

// TODO(JS): Should these be in slang-cpp-types.h?
// They are more likely to clash with slang.h

struct SlangUUID
{
    uint32_t data1;
    uint16_t data2;
    uint16_t data3;
    uint8_t data4[8];
};

typedef int32_t SlangResult;

struct ISlangUnknown
{
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    queryInterface(SlangUUID const& uuid, void** outObject) = 0;
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL addRef() = 0;
    virtual SLANG_NO_THROW uint32_t SLANG_MCALL release() = 0;
};

#define SLANG_COM_INTERFACE(a, b, c, d0, d1, d2, d3, d4, d5, d6, d7)             \
public:                                                                          \
    SLANG_FORCE_INLINE static const SlangUUID& getTypeGuid()                     \
    {                                                                            \
        static const SlangUUID guid = {a, b, c, d0, d1, d2, d3, d4, d5, d6, d7}; \
        return guid;                                                             \
    }
#endif // SLANG_H

// Includes

#ifndef SLANG_PRELUDE_SCALAR_INTRINSICS_H
#define SLANG_PRELUDE_SCALAR_INTRINSICS_H

#if !defined(SLANG_LLVM) && SLANG_PROCESSOR_X86_64 && SLANG_VC
//  If we have visual studio and 64 bit processor, we can assume we have popcnt, and can include
//  x86 intrinsics
#include <intrin.h>
#endif

#ifndef SLANG_FORCE_INLINE
#define SLANG_FORCE_INLINE inline
#endif

#ifdef SLANG_PRELUDE_NAMESPACE
namespace SLANG_PRELUDE_NAMESPACE
{
#endif

#ifndef SLANG_PRELUDE_PI
#define SLANG_PRELUDE_PI 3.14159265358979323846
#endif

union Union32
{
    uint32_t u;
    int32_t i;
    float f;
};

union Union64
{
    uint64_t u;
    int64_t i;
    double d;
};

// 32 bit cast conversions
SLANG_FORCE_INLINE int32_t _bitCastFloatToInt(float f)
{
    Union32 u;
    u.f = f;
    return u.i;
}
SLANG_FORCE_INLINE float _bitCastIntToFloat(int32_t i)
{
    Union32 u;
    u.i = i;
    return u.f;
}
SLANG_FORCE_INLINE uint32_t _bitCastFloatToUInt(float f)
{
    Union32 u;
    u.f = f;
    return u.u;
}
SLANG_FORCE_INLINE float _bitCastUIntToFloat(uint32_t ui)
{
    Union32 u;
    u.u = ui;
    return u.f;
}

// ----------------------------- F32 -----------------------------------------

// Helpers
SLANG_FORCE_INLINE float F32_calcSafeRadians(float radians);

#ifdef SLANG_LLVM

SLANG_PRELUDE_EXTERN_C_START

// Unary
float F32_ceil(float f);
float F32_floor(float f);
float F32_round(float f);
float F32_sin(float f);
float F32_cos(float f);
float F32_tan(float f);
float F32_asin(float f);
float F32_acos(float f);
float F32_atan(float f);
float F32_sinh(float f);
float F32_cosh(float f);
float F32_tanh(float f);
float F32_asinh(float f);
float F32_acosh(float f);
float F32_atanh(float f);
float F32_log2(float f);
float F32_log(float f);
float F32_log10(float f);
float F32_exp2(float f);
float F32_exp(float f);
float F32_abs(float f);
float F32_trunc(float f);
float F32_sqrt(float f);

bool F32_isnan(float f);
bool F32_isfinite(float f);
bool F32_isinf(float f);

// Binary
SLANG_FORCE_INLINE float F32_min(float a, float b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE float F32_max(float a, float b)
{
    return a > b ? a : b;
}
float F32_pow(float a, float b);
float F32_fmod(float a, float b);
float F32_remainder(float a, float b);
float F32_atan2(float a, float b);

float F32_frexp(float x, int* e);

float F32_modf(float x, float* ip);

// Ternary
SLANG_FORCE_INLINE float F32_fma(float a, float b, float c)
{
    return a * b + c;
}

SLANG_PRELUDE_EXTERN_C_END

#else

// Unary
SLANG_FORCE_INLINE float F32_ceil(float f)
{
    return ::ceilf(f);
}
SLANG_FORCE_INLINE float F32_floor(float f)
{
    return ::floorf(f);
}
SLANG_FORCE_INLINE float F32_round(float f)
{
    return ::roundf(f);
}
SLANG_FORCE_INLINE float F32_sin(float f)
{
    return ::sinf(f);
}
SLANG_FORCE_INLINE float F32_cos(float f)
{
    return ::cosf(f);
}
SLANG_FORCE_INLINE float F32_tan(float f)
{
    return ::tanf(f);
}
SLANG_FORCE_INLINE float F32_asin(float f)
{
    return ::asinf(f);
}
SLANG_FORCE_INLINE float F32_acos(float f)
{
    return ::acosf(f);
}
SLANG_FORCE_INLINE float F32_atan(float f)
{
    return ::atanf(f);
}
SLANG_FORCE_INLINE float F32_sinh(float f)
{
    return ::sinhf(f);
}
SLANG_FORCE_INLINE float F32_cosh(float f)
{
    return ::coshf(f);
}
SLANG_FORCE_INLINE float F32_tanh(float f)
{
    return ::tanhf(f);
}
SLANG_FORCE_INLINE float F32_asinh(float f)
{
    return ::asinhf(f);
}
SLANG_FORCE_INLINE float F32_acosh(float f)
{
    return ::acoshf(f);
}
SLANG_FORCE_INLINE float F32_atanh(float f)
{
    return ::atanhf(f);
}
SLANG_FORCE_INLINE float F32_log2(float f)
{
    return ::log2f(f);
}
SLANG_FORCE_INLINE float F32_log(float f)
{
    return ::logf(f);
}
SLANG_FORCE_INLINE float F32_log10(float f)
{
    return ::log10f(f);
}
SLANG_FORCE_INLINE float F32_exp2(float f)
{
    return ::exp2f(f);
}
SLANG_FORCE_INLINE float F32_exp(float f)
{
    return ::expf(f);
}
SLANG_FORCE_INLINE float F32_abs(float f)
{
    return ::fabsf(f);
}
SLANG_FORCE_INLINE float F32_trunc(float f)
{
    return ::truncf(f);
}
SLANG_FORCE_INLINE float F32_sqrt(float f)
{
    return ::sqrtf(f);
}

SLANG_FORCE_INLINE bool F32_isnan(float f)
{
    return SLANG_PRELUDE_STD isnan(f);
}
SLANG_FORCE_INLINE bool F32_isfinite(float f)
{
    return SLANG_PRELUDE_STD isfinite(f);
}
SLANG_FORCE_INLINE bool F32_isinf(float f)
{
    return SLANG_PRELUDE_STD isinf(f);
}

// Binary
SLANG_FORCE_INLINE float F32_min(float a, float b)
{
    return ::fminf(a, b);
}
SLANG_FORCE_INLINE float F32_max(float a, float b)
{
    return ::fmaxf(a, b);
}
SLANG_FORCE_INLINE float F32_pow(float a, float b)
{
    return ::powf(a, b);
}
SLANG_FORCE_INLINE float F32_fmod(float a, float b)
{
    return ::fmodf(a, b);
}
SLANG_FORCE_INLINE float F32_remainder(float a, float b)
{
    return ::remainderf(a, b);
}
SLANG_FORCE_INLINE float F32_atan2(float a, float b)
{
    return float(::atan2(a, b));
}

SLANG_FORCE_INLINE float F32_frexp(float x, int* e)
{
    return ::frexpf(x, e);
}

SLANG_FORCE_INLINE float F32_modf(float x, float* ip)
{
    return ::modff(x, ip);
}

// Ternary
SLANG_FORCE_INLINE float F32_fma(float a, float b, float c)
{
    return ::fmaf(a, b, c);
}

#endif

SLANG_FORCE_INLINE float F32_calcSafeRadians(float radians)
{
    // Put 0 to 2pi cycles to cycle around 0 to 1
    float a = radians * (1.0f / float(SLANG_PRELUDE_PI * 2));
    // Get truncated fraction, as value in  0 - 1 range
    a = a - F32_floor(a);
    // Convert back to 0 - 2pi range
    return (a * float(SLANG_PRELUDE_PI * 2));
}

SLANG_FORCE_INLINE float F32_rsqrt(float f)
{
    return 1.0f / F32_sqrt(f);
}
SLANG_FORCE_INLINE int F32_sign(float f)
{
    return (f == 0.0f) ? 0 : ((f < 0.0f) ? -1 : 1);
}
SLANG_FORCE_INLINE float F32_frac(float f)
{
    return f - F32_floor(f);
}

SLANG_FORCE_INLINE uint32_t F32_asuint(float f)
{
    Union32 u;
    u.f = f;
    return u.u;
}
SLANG_FORCE_INLINE int32_t F32_asint(float f)
{
    Union32 u;
    u.f = f;
    return u.i;
}

// ----------------------------- F64 -----------------------------------------

SLANG_FORCE_INLINE double F64_calcSafeRadians(double radians);

#ifdef SLANG_LLVM

SLANG_PRELUDE_EXTERN_C_START

// Unary
double F64_ceil(double f);
double F64_floor(double f);
double F64_round(double f);
double F64_sin(double f);
double F64_cos(double f);
double F64_tan(double f);
double F64_asin(double f);
double F64_acos(double f);
double F64_atan(double f);
double F64_sinh(double f);
double F64_cosh(double f);
double F64_tanh(double f);
double F64_asinh(double f);
double F64_acosh(double f);
double F64_atanh(double f);
double F64_log2(double f);
double F64_log(double f);
double F64_log10(double f);
double F64_exp2(double f);
double F64_exp(double f);
double F64_abs(double f);
double F64_trunc(double f);
double F64_sqrt(double f);

bool F64_isnan(double f);
bool F64_isfinite(double f);
bool F64_isinf(double f);

// Binary
SLANG_FORCE_INLINE double F64_min(double a, double b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE double F64_max(double a, double b)
{
    return a > b ? a : b;
}
double F64_pow(double a, double b);
double F64_fmod(double a, double b);
double F64_remainder(double a, double b);
double F64_atan2(double a, double b);

double F64_frexp(double x, int* e);

double F64_modf(double x, double* ip);

// Ternary
SLANG_FORCE_INLINE double F64_fma(double a, double b, double c)
{
    return a * b + c;
}

SLANG_PRELUDE_EXTERN_C_END

#else // SLANG_LLVM

// Unary
SLANG_FORCE_INLINE double F64_ceil(double f)
{
    return ::ceil(f);
}
SLANG_FORCE_INLINE double F64_floor(double f)
{
    return ::floor(f);
}
SLANG_FORCE_INLINE double F64_round(double f)
{
    return ::round(f);
}
SLANG_FORCE_INLINE double F64_sin(double f)
{
    return ::sin(f);
}
SLANG_FORCE_INLINE double F64_cos(double f)
{
    return ::cos(f);
}
SLANG_FORCE_INLINE double F64_tan(double f)
{
    return ::tan(f);
}
SLANG_FORCE_INLINE double F64_asin(double f)
{
    return ::asin(f);
}
SLANG_FORCE_INLINE double F64_acos(double f)
{
    return ::acos(f);
}
SLANG_FORCE_INLINE double F64_atan(double f)
{
    return ::atan(f);
}
SLANG_FORCE_INLINE double F64_sinh(double f)
{
    return ::sinh(f);
}
SLANG_FORCE_INLINE double F64_cosh(double f)
{
    return ::cosh(f);
}
SLANG_FORCE_INLINE double F64_tanh(double f)
{
    return ::tanh(f);
}
SLANG_FORCE_INLINE double F64_log2(double f)
{
    return ::log2(f);
}
SLANG_FORCE_INLINE double F64_log(double f)
{
    return ::log(f);
}
SLANG_FORCE_INLINE double F64_log10(float f)
{
    return ::log10(f);
}
SLANG_FORCE_INLINE double F64_exp2(double f)
{
    return ::exp2(f);
}
SLANG_FORCE_INLINE double F64_exp(double f)
{
    return ::exp(f);
}
SLANG_FORCE_INLINE double F64_abs(double f)
{
    return ::fabs(f);
}
SLANG_FORCE_INLINE double F64_trunc(double f)
{
    return ::trunc(f);
}
SLANG_FORCE_INLINE double F64_sqrt(double f)
{
    return ::sqrt(f);
}


SLANG_FORCE_INLINE bool F64_isnan(double f)
{
    return SLANG_PRELUDE_STD isnan(f);
}
SLANG_FORCE_INLINE bool F64_isfinite(double f)
{
    return SLANG_PRELUDE_STD isfinite(f);
}
SLANG_FORCE_INLINE bool F64_isinf(double f)
{
    return SLANG_PRELUDE_STD isinf(f);
}

// Binary
SLANG_FORCE_INLINE double F64_min(double a, double b)
{
    return ::fmin(a, b);
}
SLANG_FORCE_INLINE double F64_max(double a, double b)
{
    return ::fmax(a, b);
}
SLANG_FORCE_INLINE double F64_pow(double a, double b)
{
    return ::pow(a, b);
}
SLANG_FORCE_INLINE double F64_fmod(double a, double b)
{
    return ::fmod(a, b);
}
SLANG_FORCE_INLINE double F64_remainder(double a, double b)
{
    return ::remainder(a, b);
}
SLANG_FORCE_INLINE double F64_atan2(double a, double b)
{
    return ::atan2(a, b);
}

SLANG_FORCE_INLINE double F64_frexp(double x, int* e)
{
    return ::frexp(x, e);
}

SLANG_FORCE_INLINE double F64_modf(double x, double* ip)
{
    return ::modf(x, ip);
}

// Ternary
SLANG_FORCE_INLINE double F64_fma(double a, double b, double c)
{
    return ::fma(a, b, c);
}

#endif // SLANG_LLVM

SLANG_FORCE_INLINE double F64_rsqrt(double f)
{
    return 1.0 / F64_sqrt(f);
}
SLANG_FORCE_INLINE int F64_sign(double f)
{
    return (f == 0.0) ? 0 : ((f < 0.0) ? -1 : 1);
}
SLANG_FORCE_INLINE double F64_frac(double f)
{
    return f - F64_floor(f);
}

SLANG_FORCE_INLINE void F64_asuint(double d, uint32_t* low, uint32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = uint32_t(u.u);
    *hi = uint32_t(u.u >> 32);
}

SLANG_FORCE_INLINE void F64_asint(double d, int32_t* low, int32_t* hi)
{
    Union64 u;
    u.d = d;
    *low = int32_t(u.u);
    *hi = int32_t(u.u >> 32);
}

SLANG_FORCE_INLINE double F64_calcSafeRadians(double radians)
{
    // Put 0 to 2pi cycles to cycle around 0 to 1
    double a = radians * (1.0f / (SLANG_PRELUDE_PI * 2));
    // Get truncated fraction, as value in  0 - 1 range
    a = a - F64_floor(a);
    // Convert back to 0 - 2pi range
    return (a * (SLANG_PRELUDE_PI * 2));
}

// ----------------------------- F16 -----------------------------------------

// This impl is based on FloatToHalf that is in Slang codebase
SLANG_FORCE_INLINE uint32_t f32tof16(const float value)
{
    const uint32_t inBits = _bitCastFloatToUInt(value);

    // bits initially set to just the sign bit
    uint32_t bits = (inBits >> 16) & 0x8000;
    // Mantissa can't be used as is, as it holds last bit, for rounding.
    uint32_t m = (inBits >> 12) & 0x07ff;
    uint32_t e = (inBits >> 23) & 0xff;

    if (e < 103)
    {
        // It's zero
        return bits;
    }
    if (e == 0xff)
    {
        // Could be a NAN or INF. Is INF if *input* mantissa is 0.

        // Remove last bit for rounding to make output mantissa.
        m >>= 1;

        // We *assume* float16/float32 signaling bit and remaining bits
        // semantics are the same. (The signalling bit convention is target specific!).
        // Non signal bit's usage within mantissa for a NAN are also target specific.

        // If the m is 0, it could be because the result is INF, but it could also be because all
        // the bits that made NAN were dropped as we have less mantissa bits in f16.

        // To fix for this we make non zero if m is 0 and the input mantissa was not.
        // This will (typically) produce a signalling NAN.
        m += uint32_t(m == 0 && (inBits & 0x007fffffu));

        // Combine for output
        return (bits | 0x7c00u | m);
    }
    if (e > 142)
    {
        // INF.
        return bits | 0x7c00u;
    }
    if (e < 113)
    {
        m |= 0x0800u;
        bits |= (m >> (114 - e)) + ((m >> (113 - e)) & 1);
        return bits;
    }
    bits |= ((e - 112) << 10) | (m >> 1);
    bits += m & 1;
    return bits;
}

static const float g_f16tof32Magic = _bitCastIntToFloat((127 + (127 - 15)) << 23);

SLANG_FORCE_INLINE float f16tof32(const uint32_t value)
{
    const uint32_t sign = (value & 0x8000) << 16;
    uint32_t exponent = (value & 0x7c00) >> 10;
    uint32_t mantissa = (value & 0x03ff);

    if (exponent == 0)
    {
        // If mantissa is 0 we are done, as output is 0.
        // If it's not zero we must have a denormal.
        if (mantissa)
        {
            // We have a denormal so use the magic to do exponent adjust
            return _bitCastIntToFloat(sign | ((value & 0x7fff) << 13)) * g_f16tof32Magic;
        }
    }
    else
    {
        // If the exponent is NAN or INF exponent is 0x1f on input.
        // If that's the case, we just need to set the exponent to 0xff on output
        // and the mantissa can just stay the same. If its 0 it's INF, else it is NAN and we just
        // copy the bits
        //
        // Else we need to correct the exponent in the normalized case.
        exponent = (exponent == 0x1F) ? 0xff : (exponent + (-15 + 127));
    }

    return _bitCastUIntToFloat(sign | (exponent << 23) | (mantissa << 13));
}

#ifndef SLANG_LLVM
#if __cplusplus >= 202302L
#include <stdfloat> // C++23
#else
// Define __STDC_WANT_IEC_60559_TYPES_EXT__ for compilers with reliable _Float16 support:
// - Clang 15+
// - GCC 12+
#if (defined(__clang__) && __clang_major__ >= 15) || (defined(__GNUC__) && __GNUC__ >= 12)
#ifndef __STDC_WANT_IEC_60559_TYPES_EXT__
#define __STDC_WANT_IEC_60559_TYPES_EXT__
#endif
#include <float.h>
#endif // __STDC_WANT_IEC_60559_TYPES_EXT__
#endif // (defined(__clang__) && __clang_major__ >= 15) || (defined(__GNUC__) && __GNUC__ >= 12)
#endif // C++23

#ifdef FLT16_MIN
typedef _Float16 half;
#elif __STDCPP_FLOAT16_T__ == 1
typedef std::float16_t half;
#else
uint32_t f32tof16(const float value);
float f16tof32(const uint32_t value);
struct half
{
    uint16_t data;

    half() = default;
    explicit half(float f) { store(f); }

    SLANG_FORCE_INLINE void store(float f) { data = f32tof16(f); }
    SLANG_FORCE_INLINE float load() const { return f16tof32(data); }

    half operator+(half other) const { return half(load() + other.load()); }
    half operator-(half other) const { return half(load() - other.load()); }
    half operator*(half other) const { return half(load() * other.load()); }
    half operator/(half other) const { return half(load() / other.load()); }
    half& operator+=(half other)
    {
        store(load() + other.load());
        return *this;
    }
    half& operator-=(half other)
    {
        store(load() - other.load());
        return *this;
    }
    half& operator*=(half other)
    {
        store(load() * other.load());
        return *this;
    }
    half& operator/=(half other)
    {
        store(load() / other.load());
        return *this;
    }

    bool operator<(half other) const { return load() < other.load(); }
    bool operator>(half other) const { return load() > other.load(); }
    bool operator<=(half other) const { return load() <= other.load(); }
    bool operator>=(half other) const { return load() >= other.load(); }
    bool operator==(half other) const { return load() == other.load(); }
    bool operator!=(half other) const { return load() != other.load(); }

    explicit operator float() const { return load(); }
};
#endif

half U16_ashalf(uint16_t x);

union Union16
{
    uint16_t u;
    int16_t i;
    half h;
};

SLANG_FORCE_INLINE uint16_t F16_asuint(half h)
{
    Union16 u;
    u.h = h;
    return u.u;
}

SLANG_FORCE_INLINE int16_t F16_asint(half h)
{
    Union16 u;
    u.h = h;
    return u.i;
}

SLANG_FORCE_INLINE half F16_ceil(half f)
{
    return half(F32_ceil(float(f)));
}

SLANG_FORCE_INLINE half F16_floor(half f)
{
    return half(F32_floor(float(f)));
}

SLANG_FORCE_INLINE half F16_round(half f)
{
    return half(F32_round(float(f)));
}

SLANG_FORCE_INLINE half F16_sin(half f)
{
    return half(F32_sin(float(f)));
}

SLANG_FORCE_INLINE half F16_cos(half f)
{
    return half(F32_cos(float(f)));
}

SLANG_FORCE_INLINE half F16_tan(half f)
{
    return half(F32_tan(float(f)));
}

SLANG_FORCE_INLINE half F16_asin(half f)
{
    return half(F32_asin(float(f)));
}

SLANG_FORCE_INLINE half F16_acos(half f)
{
    return half(F32_acos(float(f)));
}

SLANG_FORCE_INLINE half F16_atan(half f)
{
    return half(F32_atan(float(f)));
}

SLANG_FORCE_INLINE half F16_sinh(half f)
{
    return half(F32_sinh(float(f)));
}

SLANG_FORCE_INLINE half F16_cosh(half f)
{
    return half(F32_cosh(float(f)));
}

SLANG_FORCE_INLINE half F16_tanh(half f)
{
    return half(F32_tanh(float(f)));
}

SLANG_FORCE_INLINE half F16_asinh(half f)
{
    return half(F32_asinh(float(f)));
}

SLANG_FORCE_INLINE half F16_acosh(half f)
{
    return half(F32_acosh(float(f)));
}

SLANG_FORCE_INLINE half F16_atanh(half f)
{
    return half(F32_atanh(float(f)));
}

SLANG_FORCE_INLINE half F16_log2(half f)
{
    return half(F32_log2(float(f)));
}

SLANG_FORCE_INLINE half F16_log(half f)
{
    return half(F32_log(float(f)));
}

SLANG_FORCE_INLINE half F16_log10(half f)
{
    return half(F32_log10(float(f)));
}

SLANG_FORCE_INLINE half F16_exp2(half f)
{
    return half(F32_exp2(float(f)));
}

SLANG_FORCE_INLINE half F16_exp(half f)
{
    return half(F32_exp(float(f)));
}

SLANG_FORCE_INLINE half F16_abs(half f)
{
    return U16_ashalf(F16_asuint(f) & 0x7FFF);
}

SLANG_FORCE_INLINE half F16_trunc(half f)
{
    return half(F32_trunc(float(f)));
}

SLANG_FORCE_INLINE half F16_sqrt(half f)
{
    return half(F32_sqrt(float(f)));
}

SLANG_FORCE_INLINE bool F16_isnan(half f)
{
    uint16_t u = F16_asuint(f);
    return (u & 0x7C00) == 0x7C00 && (u & 0x3FF) != 0;
}

SLANG_FORCE_INLINE bool F16_isfinite(half f)
{
    uint16_t u = F16_asuint(f);
    return (u & 0x7C00) != 0x7C00;
}

SLANG_FORCE_INLINE bool F16_isinf(half f)
{
    uint16_t u = F16_asuint(f);
    return (u & 0x7C00) == 0x7C00 && (u & 0x3FF) == 0;
}

SLANG_FORCE_INLINE half F16_min(half a, half b)
{
    if (F16_isnan(a))
        return b;
    if (F16_isnan(b))
        return a;
    return a < b ? a : b;
}

SLANG_FORCE_INLINE half F16_max(half a, half b)
{
    if (F16_isnan(a))
        return b;
    if (F16_isnan(b))
        return a;
    return a > b ? a : b;
}

SLANG_FORCE_INLINE half F16_pow(half a, half b)
{
    return half(F32_pow(float(a), float(b)));
}

SLANG_FORCE_INLINE half F16_fmod(half a, half b)
{
    return half(F32_fmod(float(a), float(b)));
}

SLANG_FORCE_INLINE half F16_remainder(half a, half b)
{
    return half(F32_remainder(float(a), float(b)));
}

SLANG_FORCE_INLINE half F16_atan2(half a, half b)
{
    return half(F32_atan2(float(a), float(b)));
}

SLANG_FORCE_INLINE half F16_frexp(half x, int* e)
{
    return half(F32_frexp(float(x), e));
}

SLANG_FORCE_INLINE half F16_modf(half x, half* ip)
{
    float ipf;
    float res = F32_modf(float(x), &ipf);
    *ip = half(ipf);
    return half(res);
}

SLANG_FORCE_INLINE half F16_fma(half a, half b, half c)
{
    return half(F32_fma(float(a), float(b), float(c)));
}

SLANG_FORCE_INLINE half F16_calcSafeRadians(half radians)
{
    // Put 0 to 2pi cycles to cycle around 0 to 1
    float a = float(radians) * (1.0f / float(SLANG_PRELUDE_PI * 2));
    // Get truncated fraction, as value in  0 - 1 range
    a = a - F32_floor(a);
    // Convert back to 0 - 2pi range
    return half(a * float(SLANG_PRELUDE_PI * 2));
}

SLANG_FORCE_INLINE half F16_rsqrt(half f)
{
    return half(1.0f / F32_sqrt(float(f)));
}

SLANG_FORCE_INLINE int F16_sign(half f)
{
    uint16_t u = F16_asuint(f);
    if ((u & 0x7FFF) == 0)
        return 0;
    return (u & 0x8000) != 0 ? -1 : 1;
}

SLANG_FORCE_INLINE half F16_frac(half h)
{
    float f = float(h);
    return half(f - F32_floor(f));
}

// ----------------------------- U16 -----------------------------------------
SLANG_FORCE_INLINE uint32_t U16_countbits(uint16_t v)
{
#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)
    return __builtin_popcount(uint32_t(v));
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    return __popcnt16(v);
#else
    uint32_t c = 0;
    while (v)
    {
        c++;
        v &= v - 1;
    }
    return c;
#endif
}

SLANG_FORCE_INLINE half U16_ashalf(uint16_t x)
{
    Union16 u;
    u.u = x;
    return u.h;
}

// ----------------------------- I16 -----------------------------------------
SLANG_FORCE_INLINE uint32_t I16_countbits(int16_t v)
{
    return U16_countbits(uint16_t(v));
}

// ----------------------------- U8 -----------------------------------------
SLANG_FORCE_INLINE uint32_t U8_countbits(uint8_t v)
{
    // No native 8bit __popcnt yet, just cast and use 16bit variant
    return U16_countbits(uint16_t(v));
}

// ----------------------------- I8 -----------------------------------------
SLANG_FORCE_INLINE uint32_t I8_countbits(int16_t v)
{
    return U8_countbits(uint8_t(v));
}

// ----------------------------- U32 -----------------------------------------

SLANG_FORCE_INLINE uint32_t U32_abs(uint32_t f)
{
    return f;
}

SLANG_FORCE_INLINE uint32_t U32_min(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE uint32_t U32_max(uint32_t a, uint32_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE float U32_asfloat(uint32_t x)
{
    Union32 u;
    u.u = x;
    return u.f;
}
SLANG_FORCE_INLINE uint32_t U32_asint(int32_t x)
{
    return uint32_t(x);
}

SLANG_FORCE_INLINE double U32_asdouble(uint32_t low, uint32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | low;
    return u.d;
}

SLANG_FORCE_INLINE uint32_t U32_countbits(uint32_t v)
{
#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)
    return __builtin_popcount(v);
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    return __popcnt(v);
#else
    uint32_t c = 0;
    while (v)
    {
        c++;
        v &= v - 1;
    }
    return c;
#endif
}

SLANG_FORCE_INLINE uint32_t U32_firstbitlow(uint32_t v)
{
    if (v == 0)
        return ~0u;

#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)
    // __builtin_ctz returns number of trailing zeros, which is the 0-based index of first set bit
    return __builtin_ctz(v);
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    // _BitScanForward returns 1 on success, 0 on failure, and sets index
    unsigned long index;
    return _BitScanForward(&index, v) ? index : ~0u;
#else
    // Generic implementation - find first set bit
    uint32_t result = 0;
    while (result < 32 && !(v & (1u << result)))
        result++;
    return result;
#endif
}

SLANG_FORCE_INLINE uint32_t U32_firstbithigh(uint32_t v)
{
    if (v == 0)
        return ~0u;
#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)
    // __builtin_clz returns number of leading zeros
    // firstbithigh should return 0-based bit position of MSB
    return 31 - __builtin_clz(v);
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    // _BitScanReverse returns 1 on success, 0 on failure, and sets index
    unsigned long index;
    return _BitScanReverse(&index, v) ? index : ~0u;
#else
    // Generic implementation - find highest set bit
    int result = 31;
    while (result >= 0 && !(v & (1u << result)))
        result--;
    return result;
#endif
}

SLANG_FORCE_INLINE uint32_t U32_reversebits(uint32_t v)
{
    v = ((v >> 1) & 0x55555555u) | ((v & 0x55555555u) << 1);
    v = ((v >> 2) & 0x33333333u) | ((v & 0x33333333u) << 2);
    v = ((v >> 4) & 0x0F0F0F0Fu) | ((v & 0x0F0F0F0Fu) << 4);
    v = ((v >> 8) & 0x00FF00FFu) | ((v & 0x00FF00FFu) << 8);
    v = (v >> 16) | (v << 16);
    return v;
}

// ----------------------------- I32 -----------------------------------------

SLANG_FORCE_INLINE int32_t I32_abs(int32_t f)
{
    return (f < 0) ? -f : f;
}

SLANG_FORCE_INLINE int32_t I32_min(int32_t a, int32_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE int32_t I32_max(int32_t a, int32_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE float I32_asfloat(int32_t x)
{
    Union32 u;
    u.i = x;
    return u.f;
}
SLANG_FORCE_INLINE uint32_t I32_asuint(int32_t x)
{
    return uint32_t(x);
}
SLANG_FORCE_INLINE double I32_asdouble(int32_t low, int32_t hi)
{
    Union64 u;
    u.u = (uint64_t(hi) << 32) | uint32_t(low);
    return u.d;
}

SLANG_FORCE_INLINE uint32_t I32_countbits(int32_t v)
{
    return U32_countbits(uint32_t(v));
}

SLANG_FORCE_INLINE uint32_t I32_firstbitlow(int32_t v)
{
    return U32_firstbitlow(uint32_t(v));
}

SLANG_FORCE_INLINE uint32_t I32_firstbithigh(int32_t v)
{
    if (v < 0)
        v = ~v;
    return U32_firstbithigh(uint32_t(v));
}

SLANG_FORCE_INLINE int32_t I32_reversebits(int32_t v)
{
    return U32_reversebits(int32_t(v));
}

// ----------------------------- U64 -----------------------------------------

SLANG_FORCE_INLINE uint64_t U64_abs(uint64_t f)
{
    return f;
}

SLANG_FORCE_INLINE uint64_t U64_min(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE uint64_t U64_max(uint64_t a, uint64_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE uint32_t U64_countbits(uint64_t v)
{
#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)
    return uint32_t(__builtin_popcountll(v));
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    return uint32_t(__popcnt64(v));
#else
    uint32_t c = 0;
    while (v)
    {
        c++;
        v &= v - 1;
    }
    return c;
#endif
}

SLANG_FORCE_INLINE uint32_t U64_firstbitlow(uint64_t v)
{
    if (v == 0)
        return ~uint32_t(0);

#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)
    // __builtin_ctz returns number of trailing zeros, which is the 0-based index of first set bit
    return __builtin_ctz(v);
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    // _BitScanForward returns 1 on success, 0 on failure, and sets index
    unsigned long index;
    return _BitScanForward64(&index, v) ? index : ~uint32_t(0);
#else
    // Generic implementation - find first set bit
    uint32_t result = 0;
    while (result < 64 && !(v & (uint64_t(1) << result)))
        result++;
    return result;
#endif
}

SLANG_FORCE_INLINE uint32_t U64_firstbithigh(uint64_t v)
{
    if (v == 0)
        return ~uint32_t(0);

#if SLANG_GCC_FAMILY && !defined(SLANG_LLVM)
    // __builtin_clz returns number of leading zeros
    // firstbithigh should return 0-based bit position of MSB
    return 63 - __builtin_clz(v);
#elif SLANG_PROCESSOR_X86_64 && SLANG_VC
    // _BitScanReverse returns 1 on success, 0 on failure, and sets index
    unsigned long index;
    return _BitScanReverse64(&index, v) ? index : ~uint32_t(0);
#else
    // Generic implementation - find highest set bit
    int result = 63;
    while (result >= 0 && !(v & (uint64_t(1) << result)))
        result--;
    return result;
#endif
}

SLANG_FORCE_INLINE uint64_t U64_reversebits(uint64_t v)
{
    v = ((v >> 1) & 0x5555555555555555ull) | ((v & 0x5555555555555555ull) << 1);
    v = ((v >> 2) & 0x3333333333333333ull) | ((v & 0x3333333333333333ull) << 2);
    v = ((v >> 4) & 0x0F0F0F0F0F0F0F0Full) | ((v & 0x0F0F0F0F0F0F0F0Full) << 4);
    v = ((v >> 8) & 0x00FF00FF00FF00FFull) | ((v & 0x00FF00FF00FF00FFull) << 8);
    v = ((v >> 16) & 0x0000FFFF0000FFFFull) | ((v & 0x0000FFFF0000FFFFull) << 16);
    v = (v >> 32) | (v << 32);
    return v;
}

// ----------------------------- I64 -----------------------------------------

SLANG_FORCE_INLINE int64_t I64_abs(int64_t f)
{
    return (f < 0) ? -f : f;
}

SLANG_FORCE_INLINE int64_t I64_min(int64_t a, int64_t b)
{
    return a < b ? a : b;
}
SLANG_FORCE_INLINE int64_t I64_max(int64_t a, int64_t b)
{
    return a > b ? a : b;
}

SLANG_FORCE_INLINE uint32_t I64_countbits(int64_t v)
{
    return U64_countbits(uint64_t(v));
}

SLANG_FORCE_INLINE uint32_t I64_firstbitlow(int64_t v)
{
    return U64_firstbitlow(uint64_t(v));
}

SLANG_FORCE_INLINE uint32_t I64_firstbithigh(int64_t v)
{
    if (v < 0)
        v = ~v;
    return U64_firstbithigh(uint64_t(v));
}

SLANG_FORCE_INLINE int64_t I64_reversebits(int64_t v)
{
    return int64_t(U64_reversebits(uint64_t(v)));
}

// ----------------------------- UPTR -----------------------------------------

SLANG_FORCE_INLINE uintptr_t UPTR_abs(uintptr_t f)
{
    return f;
}

SLANG_FORCE_INLINE uintptr_t UPTR_min(uintptr_t a, uintptr_t b)
{
    return a < b ? a : b;
}

SLANG_FORCE_INLINE uintptr_t UPTR_max(uintptr_t a, uintptr_t b)
{
    return a > b ? a : b;
}

// ----------------------------- IPTR -----------------------------------------

SLANG_FORCE_INLINE intptr_t IPTR_abs(intptr_t f)
{
    return (f < 0) ? -f : f;
}

SLANG_FORCE_INLINE intptr_t IPTR_min(intptr_t a, intptr_t b)
{
    return a < b ? a : b;
}

SLANG_FORCE_INLINE intptr_t IPTR_max(intptr_t a, intptr_t b)
{
    return a > b ? a : b;
}

// ----------------------------- Interlocked ---------------------------------

#if SLANG_LLVM

#else // SLANG_LLVM

#ifdef _WIN32
#include <intrin.h>
#endif

SLANG_FORCE_INLINE void InterlockedAdd(uint32_t* dest, uint32_t value, uint32_t* oldValue)
{
#ifdef _WIN32
    *oldValue = _InterlockedExchangeAdd((long*)dest, (long)value);
#else
    *oldValue = __sync_fetch_and_add(dest, value);
#endif
}

#endif // SLANG_LLVM


// ----------------------- fmod --------------------------
SLANG_FORCE_INLINE float _slang_fmod(float x, float y)
{
    return F32_fmod(x, y);
}
SLANG_FORCE_INLINE double _slang_fmod(double x, double y)
{
    return F64_fmod(x, y);
}

#ifdef SLANG_PRELUDE_NAMESPACE
}
#endif

#endif

#ifndef SLANG_PRELUDE_CPP_TYPES_H
#define SLANG_PRELUDE_CPP_TYPES_H

#ifdef SLANG_PRELUDE_NAMESPACE
namespace SLANG_PRELUDE_NAMESPACE
{
#endif

#ifndef SLANG_FORCE_INLINE
#define SLANG_FORCE_INLINE inline
#endif

#ifndef SLANG_PRELUDE_CPP_TYPES_CORE_H
#define SLANG_PRELUDE_CPP_TYPES_CORE_H

#ifndef SLANG_PRELUDE_ASSERT
#ifdef SLANG_PRELUDE_ENABLE_ASSERT
#define SLANG_PRELUDE_ASSERT(VALUE) assert(VALUE)
#else
#define SLANG_PRELUDE_ASSERT(VALUE)
#endif
#endif

// Since we are using unsigned arithmatic care is need in this comparison.
// It is *assumed* that sizeInBytes >= elemSize. Which means (sizeInBytes >= elemSize) >= 0
// Which means only a single test is needed

// Asserts for bounds checking.
// It is assumed index/count are unsigned types.
#define SLANG_BOUND_ASSERT(index, count) SLANG_PRELUDE_ASSERT(index < count);
#define SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_PRELUDE_ASSERT(index <= (sizeInBytes - elemSize) && (index & 3) == 0);

// Macros to zero index if an access is out of range
#define SLANG_BOUND_ZERO_INDEX(index, count) index = (index < count) ? index : 0;
#define SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    index = (index <= (sizeInBytes - elemSize)) ? index : 0;

// The 'FIX' macro define how the index is fixed. The default is to do nothing. If
// SLANG_ENABLE_BOUND_ZERO_INDEX the fix macro will zero the index, if out of range
#ifdef SLANG_ENABLE_BOUND_ZERO_INDEX
#define SLANG_BOUND_FIX(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ZERO_INDEX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count) SLANG_BOUND_ZERO_INDEX(index, count)
#else
#define SLANG_BOUND_FIX(index, count)
#define SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#define SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

#ifndef SLANG_BOUND_CHECK
#define SLANG_BOUND_CHECK(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX(index, count)
#endif

#ifndef SLANG_BOUND_CHECK_BYTE_ADDRESS
#define SLANG_BOUND_CHECK_BYTE_ADDRESS(index, elemSize, sizeInBytes) \
    SLANG_BOUND_ASSERT_BYTE_ADDRESS(index, elemSize, sizeInBytes)    \
    SLANG_BOUND_FIX_BYTE_ADDRESS(index, elemSize, sizeInBytes)
#endif

#ifndef SLANG_BOUND_CHECK_FIXED_ARRAY
#define SLANG_BOUND_CHECK_FIXED_ARRAY(index, count) \
    SLANG_BOUND_ASSERT(index, count) SLANG_BOUND_FIX_FIXED_ARRAY(index, count)
#endif

struct TypeInfo
{
    size_t typeSize;
};

template<typename T, size_t SIZE>
struct FixedArray
{
    const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }
    T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK_FIXED_ARRAY(index, SIZE);
        return m_data[index];
    }

    T m_data[SIZE];
};

// An array that has no specified size, becomes a 'Array'. This stores the size so it can
// potentially do bounds checking.
template<typename T>
struct Array
{
    const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    T& operator[](size_t index)
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }

    T* data;
    size_t count;
};

/* Constant buffers become a pointer to the contained type, so ConstantBuffer<T> becomes T* in C++
 * code.
 */

template<typename T, int COUNT>
struct Vector;

template<typename T>
struct Vector<T, 1>
{
    T x;
    const T& operator[](size_t /*index*/) const { return x; }
    T& operator[](size_t /*index*/) { return x; }
    operator T() const { return x; }
    Vector() = default;
    Vector(T scalar) { x = scalar; }
    template<typename U>
    Vector(Vector<U, 1> other)
    {
        x = (T)other.x;
    }
    template<typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 1;
        if (otherSize < minSize)
            minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
};

template<typename T>
struct Vector<T, 2>
{
    T x, y;
    const T& operator[](size_t index) const { return index == 0 ? x : y; }
    T& operator[](size_t index) { return index == 0 ? x : y; }
    Vector() = default;
    Vector(T scalar) { x = y = scalar; }
    Vector(T _x, T _y)
    {
        x = _x;
        y = _y;
    }
    template<typename U>
    Vector(Vector<U, 2> other)
    {
        x = (T)other.x;
        y = (T)other.y;
    }
    template<typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 2;
        if (otherSize < minSize)
            minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
};

template<typename T>
struct Vector<T, 3>
{
    T x, y, z;
    const T& operator[](size_t index) const { return *((T*)(this) + index); }
    T& operator[](size_t index) { return *((T*)(this) + index); }

    Vector() = default;
    Vector(T scalar) { x = y = z = scalar; }
    Vector(T _x, T _y, T _z)
    {
        x = _x;
        y = _y;
        z = _z;
    }
    template<typename U>
    Vector(Vector<U, 3> other)
    {
        x = (T)other.x;
        y = (T)other.y;
        z = (T)other.z;
    }
    template<typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 3;
        if (otherSize < minSize)
            minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
};

template<typename T>
struct Vector<T, 4>
{
    T x, y, z, w;

    const T& operator[](size_t index) const { return *((T*)(this) + index); }
    T& operator[](size_t index) { return *((T*)(this) + index); }
    Vector() = default;
    Vector(T scalar) { x = y = z = w = scalar; }
    Vector(T _x, T _y, T _z, T _w)
    {
        x = _x;
        y = _y;
        z = _z;
        w = _w;
    }
    template<typename U, int otherSize>
    Vector(Vector<U, otherSize> other)
    {
        int minSize = 4;
        if (otherSize < minSize)
            minSize = otherSize;
        for (int i = 0; i < minSize; i++)
            (*this)[i] = (T)other[i];
    }
};

template<typename T, int N>
SLANG_FORCE_INLINE Vector<T, N> _slang_select(
    Vector<bool, N> condition,
    Vector<T, N> v0,
    Vector<T, N> v1)
{
    Vector<T, N> result;
    for (int i = 0; i < N; i++)
    {
        result[i] = condition[i] ? v0[i] : v1[i];
    }
    return result;
}

template<typename T>
SLANG_FORCE_INLINE T _slang_select(bool condition, T v0, T v1)
{
    return condition ? v0 : v1;
}

template<typename T, int N>
SLANG_FORCE_INLINE T _slang_vector_get_element(Vector<T, N> x, int index)
{
    return x[index];
}

template<typename T, int N>
SLANG_FORCE_INLINE const T* _slang_vector_get_element_ptr(const Vector<T, N>* x, int index)
{
    return &((*const_cast<Vector<T, N>*>(x))[index]);
}

template<typename T, int N>
SLANG_FORCE_INLINE T* _slang_vector_get_element_ptr(Vector<T, N>* x, int index)
{
    return &((*x)[index]);
}

template<typename T, int n, typename OtherT, int m>
SLANG_FORCE_INLINE Vector<T, n> _slang_vector_reshape(const Vector<OtherT, m> other)
{
    Vector<T, n> result;
    for (int i = 0; i < n; i++)
    {
        OtherT otherElement = T(0);
        if (i < m)
            otherElement = _slang_vector_get_element(other, i);
        *_slang_vector_get_element_ptr(&result, i) = (T)otherElement;
    }
    return result;
}

typedef uint32_t uint;

#define SLANG_VECTOR_BINARY_OP(T, op)            \
    template<int n>                              \
    SLANG_FORCE_INLINE Vector<T, n> operator op( \
        const Vector<T, n>& thisVal,             \
        const Vector<T, n>& other)               \
    {                                            \
        Vector<T, n> result;                     \
        for (int i = 0; i < n; i++)              \
            result[i] = thisVal[i] op other[i];  \
        return result;                           \
    }
#define SLANG_VECTOR_BINARY_COMPARE_OP(T, op)       \
    template<int n>                                 \
    SLANG_FORCE_INLINE Vector<bool, n> operator op( \
        const Vector<T, n>& thisVal,                \
        const Vector<T, n>& other)                  \
    {                                               \
        Vector<bool, n> result;                     \
        for (int i = 0; i < n; i++)                 \
            result[i] = thisVal[i] op other[i];     \
        return result;                              \
    }

#define SLANG_VECTOR_UNARY_OP(T, op)                                         \
    template<int n>                                                          \
    SLANG_FORCE_INLINE Vector<T, n> operator op(const Vector<T, n>& thisVal) \
    {                                                                        \
        Vector<T, n> result;                                                 \
        for (int i = 0; i < n; i++)                                          \
            result[i] = op thisVal[i];                                       \
        return result;                                                       \
    }
#define SLANG_INT_VECTOR_OPS(T)           \
    SLANG_VECTOR_BINARY_OP(T, +)          \
    SLANG_VECTOR_BINARY_OP(T, -)          \
    SLANG_VECTOR_BINARY_OP(T, *)          \
    SLANG_VECTOR_BINARY_OP(T, /)          \
    SLANG_VECTOR_BINARY_OP(T, &)          \
    SLANG_VECTOR_BINARY_OP(T, |)          \
    SLANG_VECTOR_BINARY_OP(T, &&)         \
    SLANG_VECTOR_BINARY_OP(T, ||)         \
    SLANG_VECTOR_BINARY_OP(T, ^)          \
    SLANG_VECTOR_BINARY_OP(T, %)          \
    SLANG_VECTOR_BINARY_OP(T, >>)         \
    SLANG_VECTOR_BINARY_OP(T, <<)         \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >)  \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <)  \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >=) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <=) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, ==) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, !=) \
    SLANG_VECTOR_UNARY_OP(T, !)           \
    SLANG_VECTOR_UNARY_OP(T, ~)
#define SLANG_FLOAT_VECTOR_OPS(T)         \
    SLANG_VECTOR_BINARY_OP(T, +)          \
    SLANG_VECTOR_BINARY_OP(T, -)          \
    SLANG_VECTOR_BINARY_OP(T, *)          \
    SLANG_VECTOR_BINARY_OP(T, /)          \
    SLANG_VECTOR_UNARY_OP(T, -)           \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >)  \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <)  \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, >=) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, <=) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, ==) \
    SLANG_VECTOR_BINARY_COMPARE_OP(T, !=)

SLANG_INT_VECTOR_OPS(bool)
SLANG_INT_VECTOR_OPS(int)
SLANG_INT_VECTOR_OPS(int8_t)
SLANG_INT_VECTOR_OPS(int16_t)
SLANG_INT_VECTOR_OPS(int64_t)
SLANG_INT_VECTOR_OPS(uint)
SLANG_INT_VECTOR_OPS(uint8_t)
SLANG_INT_VECTOR_OPS(uint16_t)
SLANG_INT_VECTOR_OPS(uint64_t)
#if SLANG_INTPTR_TYPE_IS_DISTINCT
SLANG_INT_VECTOR_OPS(intptr_t)
SLANG_INT_VECTOR_OPS(uintptr_t)
#endif

SLANG_FLOAT_VECTOR_OPS(float)
SLANG_FLOAT_VECTOR_OPS(double)

#define SLANG_VECTOR_INT_NEG_OP(T)                      \
    template<int N>                                     \
    Vector<T, N> operator-(const Vector<T, N>& thisVal) \
    {                                                   \
        Vector<T, N> result;                            \
        for (int i = 0; i < N; i++)                     \
            result[i] = 0 - thisVal[i];                 \
        return result;                                  \
    }
SLANG_VECTOR_INT_NEG_OP(int)
SLANG_VECTOR_INT_NEG_OP(int8_t)
SLANG_VECTOR_INT_NEG_OP(int16_t)
SLANG_VECTOR_INT_NEG_OP(int64_t)
SLANG_VECTOR_INT_NEG_OP(uint)
SLANG_VECTOR_INT_NEG_OP(uint8_t)
SLANG_VECTOR_INT_NEG_OP(uint16_t)
SLANG_VECTOR_INT_NEG_OP(uint64_t)
#if SLANG_INTPTR_TYPE_IS_DISTINCT
SLANG_VECTOR_INT_NEG_OP(intptr_t)
SLANG_VECTOR_INT_NEG_OP(uintptr_t)
#endif

#define SLANG_FLOAT_VECTOR_MOD(T)                                               \
    template<int N>                                                             \
    Vector<T, N> operator%(const Vector<T, N>& left, const Vector<T, N>& right) \
    {                                                                           \
        Vector<T, N> result;                                                    \
        for (int i = 0; i < N; i++)                                             \
            result[i] = _slang_fmod(left[i], right[i]);                         \
        return result;                                                          \
    }

SLANG_FLOAT_VECTOR_MOD(float)
SLANG_FLOAT_VECTOR_MOD(double)
#undef SLANG_FLOAT_VECTOR_MOD
#undef SLANG_VECTOR_BINARY_OP
#undef SLANG_VECTOR_UNARY_OP
#undef SLANG_INT_VECTOR_OPS
#undef SLANG_FLOAT_VECTOR_OPS
#undef SLANG_VECTOR_INT_NEG_OP
#undef SLANG_FLOAT_VECTOR_MOD

template<typename T, int ROWS, int COLS>
struct Matrix
{
    Vector<T, COLS> rows[ROWS];
    const Vector<T, COLS>& operator[](size_t index) const { return rows[index]; }
    Vector<T, COLS>& operator[](size_t index) { return rows[index]; }
    Matrix() = default;
    Matrix(T scalar)
    {
        for (int i = 0; i < ROWS; i++)
            rows[i] = Vector<T, COLS>(scalar);
    }
    Matrix(const Vector<T, COLS>& row0) { rows[0] = row0; }
    Matrix(const Vector<T, COLS>& row0, const Vector<T, COLS>& row1)
    {
        rows[0] = row0;
        rows[1] = row1;
    }
    Matrix(const Vector<T, COLS>& row0, const Vector<T, COLS>& row1, const Vector<T, COLS>& row2)
    {
        rows[0] = row0;
        rows[1] = row1;
        rows[2] = row2;
    }
    Matrix(
        const Vector<T, COLS>& row0,
        const Vector<T, COLS>& row1,
        const Vector<T, COLS>& row2,
        const Vector<T, COLS>& row3)
    {
        rows[0] = row0;
        rows[1] = row1;
        rows[2] = row2;
        rows[3] = row3;
    }
    template<typename U, int otherRow, int otherCol>
    Matrix(const Matrix<U, otherRow, otherCol>& other)
    {
        int minRow = ROWS;
        int minCol = COLS;
        if (minRow > otherRow)
            minRow = otherRow;
        if (minCol > otherCol)
            minCol = otherCol;
        for (int i = 0; i < minRow; i++)
            for (int j = 0; j < minCol; j++)
                rows[i][j] = (T)other.rows[i][j];
    }
    Matrix(T v0, T v1, T v2, T v3)
    {
        rows[0][0] = v0;
        rows[0][1] = v1;
        rows[1][0] = v2;
        rows[1][1] = v3;
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5)
    {
        if (COLS == 3)
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[0][2] = v2;
            rows[1][0] = v3;
            rows[1][1] = v4;
            rows[1][2] = v5;
        }
        else
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[1][0] = v2;
            rows[1][1] = v3;
            rows[2][0] = v4;
            rows[2][1] = v5;
        }
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7)
    {
        if (COLS == 4)
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[0][2] = v2;
            rows[0][3] = v3;
            rows[1][0] = v4;
            rows[1][1] = v5;
            rows[1][2] = v6;
            rows[1][3] = v7;
        }
        else
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[1][0] = v2;
            rows[1][1] = v3;
            rows[2][0] = v4;
            rows[2][1] = v5;
            rows[3][0] = v6;
            rows[3][1] = v7;
        }
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8)
    {
        rows[0][0] = v0;
        rows[0][1] = v1;
        rows[0][2] = v2;
        rows[1][0] = v3;
        rows[1][1] = v4;
        rows[1][2] = v5;
        rows[2][0] = v6;
        rows[2][1] = v7;
        rows[2][2] = v8;
    }
    Matrix(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7, T v8, T v9, T v10, T v11)
    {
        if (COLS == 4)
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[0][2] = v2;
            rows[0][3] = v3;
            rows[1][0] = v4;
            rows[1][1] = v5;
            rows[1][2] = v6;
            rows[1][3] = v7;
            rows[2][0] = v8;
            rows[2][1] = v9;
            rows[2][2] = v10;
            rows[2][3] = v11;
        }
        else
        {
            rows[0][0] = v0;
            rows[0][1] = v1;
            rows[0][2] = v2;
            rows[1][0] = v3;
            rows[1][1] = v4;
            rows[1][2] = v5;
            rows[2][0] = v6;
            rows[2][1] = v7;
            rows[2][2] = v8;
            rows[3][0] = v9;
            rows[3][1] = v10;
            rows[3][2] = v11;
        }
    }
    Matrix(
        T v0,
        T v1,
        T v2,
        T v3,
        T v4,
        T v5,
        T v6,
        T v7,
        T v8,
        T v9,
        T v10,
        T v11,
        T v12,
        T v13,
        T v14,
        T v15)
    {
        rows[0][0] = v0;
        rows[0][1] = v1;
        rows[0][2] = v2;
        rows[0][3] = v3;
        rows[1][0] = v4;
        rows[1][1] = v5;
        rows[1][2] = v6;
        rows[1][3] = v7;
        rows[2][0] = v8;
        rows[2][1] = v9;
        rows[2][2] = v10;
        rows[2][3] = v11;
        rows[3][0] = v12;
        rows[3][1] = v13;
        rows[3][2] = v14;
        rows[3][3] = v15;
    }
};

#define SLANG_MATRIX_BINARY_OP(T, op)                                                         \
    template<int R, int C>                                                                    \
    Matrix<T, R, C> operator op(const Matrix<T, R, C>& thisVal, const Matrix<T, R, C>& other) \
    {                                                                                         \
        Matrix<T, R, C> result;                                                               \
        for (int i = 0; i < R; i++)                                                           \
            for (int j = 0; j < C; j++)                                                       \
                result.rows[i][j] = thisVal.rows[i][j] op other.rows[i][j];                   \
        return result;                                                                        \
    }

#define SLANG_MATRIX_BINARY_COMPARE_OP(T, op)                                                    \
    template<int R, int C>                                                                       \
    Matrix<bool, R, C> operator op(const Matrix<T, R, C>& thisVal, const Matrix<T, R, C>& other) \
    {                                                                                            \
        Matrix<bool, R, C> result;                                                               \
        for (int i = 0; i < R; i++)                                                              \
            for (int j = 0; j < C; j++)                                                          \
                result.rows[i][j] = thisVal.rows[i][j] op other.rows[i][j];                      \
        return result;                                                                           \
    }

#define SLANG_MATRIX_UNARY_OP(T, op)                            \
    template<int R, int C>                                      \
    Matrix<T, R, C> operator op(const Matrix<T, R, C>& thisVal) \
    {                                                           \
        Matrix<T, R, C> result;                                 \
        for (int i = 0; i < R; i++)                             \
            for (int j = 0; j < C; j++)                         \
                result[i].rows[i][j] = op thisVal.rows[i][j];   \
        return result;                                          \
    }

#define SLANG_INT_MATRIX_OPS(T)           \
    SLANG_MATRIX_BINARY_OP(T, +)          \
    SLANG_MATRIX_BINARY_OP(T, -)          \
    SLANG_MATRIX_BINARY_OP(T, *)          \
    SLANG_MATRIX_BINARY_OP(T, /)          \
    SLANG_MATRIX_BINARY_OP(T, &)          \
    SLANG_MATRIX_BINARY_OP(T, |)          \
    SLANG_MATRIX_BINARY_OP(T, &&)         \
    SLANG_MATRIX_BINARY_OP(T, ||)         \
    SLANG_MATRIX_BINARY_OP(T, ^)          \
    SLANG_MATRIX_BINARY_OP(T, %)          \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, >)  \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, <)  \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, >=) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, <=) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, ==) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, !=) \
    SLANG_MATRIX_UNARY_OP(T, !)           \
    SLANG_MATRIX_UNARY_OP(T, ~)
#define SLANG_FLOAT_MATRIX_OPS(T)         \
    SLANG_MATRIX_BINARY_OP(T, +)          \
    SLANG_MATRIX_BINARY_OP(T, -)          \
    SLANG_MATRIX_BINARY_OP(T, *)          \
    SLANG_MATRIX_BINARY_OP(T, /)          \
    SLANG_MATRIX_UNARY_OP(T, -)           \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, >)  \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, <)  \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, >=) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, <=) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, ==) \
    SLANG_MATRIX_BINARY_COMPARE_OP(T, !=)
SLANG_INT_MATRIX_OPS(int)
SLANG_INT_MATRIX_OPS(int8_t)
SLANG_INT_MATRIX_OPS(int16_t)
SLANG_INT_MATRIX_OPS(int64_t)
SLANG_INT_MATRIX_OPS(uint)
SLANG_INT_MATRIX_OPS(uint8_t)
SLANG_INT_MATRIX_OPS(uint16_t)
SLANG_INT_MATRIX_OPS(uint64_t)
#if SLANG_INTPTR_TYPE_IS_DISTINCT
SLANG_INT_MATRIX_OPS(intptr_t)
SLANG_INT_MATRIX_OPS(uintptr_t)
#endif

SLANG_FLOAT_MATRIX_OPS(float)
SLANG_FLOAT_MATRIX_OPS(double)

#define SLANG_MATRIX_INT_NEG_OP(T)                                        \
    template<int R, int C>                                                \
    SLANG_FORCE_INLINE Matrix<T, R, C> operator-(Matrix<T, R, C> thisVal) \
    {                                                                     \
        Matrix<T, R, C> result;                                           \
        for (int i = 0; i < R; i++)                                       \
            for (int j = 0; j < C; j++)                                   \
                result.rows[i][j] = 0 - thisVal.rows[i][j];               \
        return result;                                                    \
    }
SLANG_MATRIX_INT_NEG_OP(int)
SLANG_MATRIX_INT_NEG_OP(int8_t)
SLANG_MATRIX_INT_NEG_OP(int16_t)
SLANG_MATRIX_INT_NEG_OP(int64_t)
SLANG_MATRIX_INT_NEG_OP(uint)
SLANG_MATRIX_INT_NEG_OP(uint8_t)
SLANG_MATRIX_INT_NEG_OP(uint16_t)
SLANG_MATRIX_INT_NEG_OP(uint64_t)
#if SLANG_INTPTR_TYPE_IS_DISTINCT
SLANG_MATRIX_INT_NEG_OP(intptr_t)
SLANG_MATRIX_INT_NEG_OP(uintptr_t)
#endif

#define SLANG_FLOAT_MATRIX_MOD(T)                                                             \
    template<int R, int C>                                                                    \
    SLANG_FORCE_INLINE Matrix<T, R, C> operator%(Matrix<T, R, C> left, Matrix<T, R, C> right) \
    {                                                                                         \
        Matrix<T, R, C> result;                                                               \
        for (int i = 0; i < R; i++)                                                           \
            for (int j = 0; j < C; j++)                                                       \
                result.rows[i][j] = _slang_fmod(left.rows[i][j], right.rows[i][j]);           \
        return result;                                                                        \
    }

SLANG_FLOAT_MATRIX_MOD(float)
SLANG_FLOAT_MATRIX_MOD(double)
#undef SLANG_FLOAT_MATRIX_MOD
#undef SLANG_MATRIX_BINARY_OP
#undef SLANG_MATRIX_UNARY_OP
#undef SLANG_INT_MATRIX_OPS
#undef SLANG_FLOAT_MATRIX_OPS
#undef SLANG_MATRIX_INT_NEG_OP
#undef SLANG_FLOAT_MATRIX_MOD

template<typename TResult, typename TInput>
TResult slang_bit_cast(TInput val)
{
    return *(TResult*)(&val);
}

#endif


typedef Vector<float, 2> float2;
typedef Vector<float, 3> float3;
typedef Vector<float, 4> float4;

typedef Vector<int32_t, 2> int2;
typedef Vector<int32_t, 3> int3;
typedef Vector<int32_t, 4> int4;

typedef Vector<uint32_t, 2> uint2;
typedef Vector<uint32_t, 3> uint3;
typedef Vector<uint32_t, 4> uint4;

// We can just map `NonUniformResourceIndex` type directly to the index type on CPU, as CPU does not
// require any special handling around such accesses.
typedef size_t NonUniformResourceIndex;

// ----------------------------- ResourceType -----------------------------------------

// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-structuredbuffer-getdimensions
// Missing  Load(_In_  int  Location, _Out_ uint Status);

template<typename T>
struct RWStructuredBuffer
{
    SLANG_FORCE_INLINE T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    const T& Load(size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    void GetDimensions(uint32_t* outNumStructs, uint32_t* outStride)
    {
        *outNumStructs = uint32_t(count);
        *outStride = uint32_t(sizeof(T));
    }

    T* data;
    size_t count;
};

template<typename T>
struct StructuredBuffer
{
    SLANG_FORCE_INLINE T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    T& Load(size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    void GetDimensions(uint32_t* outNumStructs, uint32_t* outStride)
    {
        *outNumStructs = uint32_t(count);
        *outStride = uint32_t(sizeof(T));
    }

    T* data;
    size_t count;
};


template<typename T>
struct RWBuffer
{
    SLANG_FORCE_INLINE T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    const T& Load(size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    void GetDimensions(uint32_t* outCount) { *outCount = uint32_t(count); }

    T* data;
    size_t count;
};

template<typename T>
struct Buffer
{
    SLANG_FORCE_INLINE const T& operator[](size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    const T& Load(size_t index) const
    {
        SLANG_BOUND_CHECK(index, count);
        return data[index];
    }
    void GetDimensions(uint32_t* outCount) { *outCount = uint32_t(count); }

    T* data;
    size_t count;
};

// Missing  Load(_In_  int  Location, _Out_ uint Status);
struct ByteAddressBuffer
{
    void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }
    uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        return *(const T*)(((const char*)data) + index);
    }

    const uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};

// https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/sm5-object-rwbyteaddressbuffer
// Missing support for Atomic operations
// Missing support for Load with status
struct RWByteAddressBuffer
{
    void GetDimensions(uint32_t* outDim) const { *outDim = uint32_t(sizeInBytes); }

    uint32_t Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        return data[index >> 2];
    }
    uint2 Load2(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint2{data[dataIdx], data[dataIdx + 1]};
    }
    uint3 Load3(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint3{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2]};
    }
    uint4 Load4(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        return uint4{data[dataIdx], data[dataIdx + 1], data[dataIdx + 2], data[dataIdx + 3]};
    }
    template<typename T>
    T Load(size_t index) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        return *(const T*)(((const char*)data) + index);
    }

    void Store(size_t index, uint32_t v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 4, sizeInBytes);
        data[index >> 2] = v;
    }
    void Store2(size_t index, uint2 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 8, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
    }
    void Store3(size_t index, uint3 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 12, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
    }
    void Store4(size_t index, uint4 v) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, 16, sizeInBytes);
        const size_t dataIdx = index >> 2;
        data[dataIdx + 0] = v.x;
        data[dataIdx + 1] = v.y;
        data[dataIdx + 2] = v.z;
        data[dataIdx + 3] = v.w;
    }
    template<typename T>
    void Store(size_t index, T const& value) const
    {
        SLANG_BOUND_CHECK_BYTE_ADDRESS(index, sizeof(T), sizeInBytes);
        *(T*)(((char*)data) + index) = value;
    }

    uint32_t* data;
    size_t sizeInBytes; //< Must be multiple of 4
};

struct ISamplerState;
struct ISamplerComparisonState;

struct SamplerState
{
    ISamplerState* state;
};

struct SamplerComparisonState
{
    ISamplerComparisonState* state;
};

#ifndef SLANG_RESOURCE_SHAPE
#define SLANG_RESOURCE_SHAPE
typedef unsigned int SlangResourceShape;
enum
{
    SLANG_RESOURCE_BASE_SHAPE_MASK = 0x0F,

    SLANG_RESOURCE_NONE = 0x00,

    SLANG_TEXTURE_1D = 0x01,
    SLANG_TEXTURE_2D = 0x02,
    SLANG_TEXTURE_3D = 0x03,
    SLANG_TEXTURE_CUBE = 0x04,
    SLANG_TEXTURE_BUFFER = 0x05,

    SLANG_STRUCTURED_BUFFER = 0x06,
    SLANG_BYTE_ADDRESS_BUFFER = 0x07,
    SLANG_RESOURCE_UNKNOWN = 0x08,
    SLANG_ACCELERATION_STRUCTURE = 0x09,
    SLANG_TEXTURE_SUBPASS = 0x0A,

    SLANG_RESOURCE_EXT_SHAPE_MASK = 0xF0,

    SLANG_TEXTURE_FEEDBACK_FLAG = 0x10,
    SLANG_TEXTURE_ARRAY_FLAG = 0x40,
    SLANG_TEXTURE_MULTISAMPLE_FLAG = 0x80,

    SLANG_TEXTURE_1D_ARRAY = SLANG_TEXTURE_1D | SLANG_TEXTURE_ARRAY_FLAG,
    SLANG_TEXTURE_2D_ARRAY = SLANG_TEXTURE_2D | SLANG_TEXTURE_ARRAY_FLAG,
    SLANG_TEXTURE_CUBE_ARRAY = SLANG_TEXTURE_CUBE | SLANG_TEXTURE_ARRAY_FLAG,

    SLANG_TEXTURE_2D_MULTISAMPLE = SLANG_TEXTURE_2D | SLANG_TEXTURE_MULTISAMPLE_FLAG,
    SLANG_TEXTURE_2D_MULTISAMPLE_ARRAY =
        SLANG_TEXTURE_2D | SLANG_TEXTURE_MULTISAMPLE_FLAG | SLANG_TEXTURE_ARRAY_FLAG,
    SLANG_TEXTURE_SUBPASS_MULTISAMPLE = SLANG_TEXTURE_SUBPASS | SLANG_TEXTURE_MULTISAMPLE_FLAG,
};
#endif

//
struct TextureDimensions
{
    void reset()
    {
        shape = 0;
        width = height = depth = 0;
        numberOfLevels = 0;
        arrayElementCount = 0;
    }
    int getDimSizes(uint32_t outDims[4]) const
    {
        const auto baseShape = (shape & SLANG_RESOURCE_BASE_SHAPE_MASK);
        int count = 0;
        switch (baseShape)
        {
        case SLANG_TEXTURE_1D:
            {
                outDims[count++] = width;
                break;
            }
        case SLANG_TEXTURE_2D:
            {
                outDims[count++] = width;
                outDims[count++] = height;
                break;
            }
        case SLANG_TEXTURE_3D:
            {
                outDims[count++] = width;
                outDims[count++] = height;
                outDims[count++] = depth;
                break;
            }
        case SLANG_TEXTURE_CUBE:
            {
                outDims[count++] = width;
                outDims[count++] = height;
                outDims[count++] = 6;
                break;
            }
        }

        if (shape & SLANG_TEXTURE_ARRAY_FLAG)
        {
            outDims[count++] = arrayElementCount;
        }
        return count;
    }
    int getMIPDims(int outDims[3]) const
    {
        const auto baseShape = (shape & SLANG_RESOURCE_BASE_SHAPE_MASK);
        int count = 0;
        switch (baseShape)
        {
        case SLANG_TEXTURE_1D:
            {
                outDims[count++] = width;
                break;
            }
        case SLANG_TEXTURE_CUBE:
        case SLANG_TEXTURE_2D:
            {
                outDims[count++] = width;
                outDims[count++] = height;
                break;
            }
        case SLANG_TEXTURE_3D:
            {
                outDims[count++] = width;
                outDims[count++] = height;
                outDims[count++] = depth;
                break;
            }
        }
        return count;
    }
    int calcMaxMIPLevels() const
    {
        int dims[3];
        const int dimCount = getMIPDims(dims);
        for (int count = 1; true; count++)
        {
            bool allOne = true;
            for (int i = 0; i < dimCount; ++i)
            {
                if (dims[i] > 1)
                {
                    allOne = false;
                    dims[i] >>= 1;
                }
            }
            if (allOne)
            {
                return count;
            }
        }
    }

    uint32_t shape;
    uint32_t width, height, depth;
    uint32_t numberOfLevels;
    uint32_t arrayElementCount; ///< For array types, 0 otherwise
};


// Texture

struct ITexture
{
    virtual TextureDimensions GetDimensions(int mipLevel = -1) = 0;
    virtual void Load(const int32_t* v, void* outData, size_t dataSize) = 0;
    virtual void Sample(
        SamplerState samplerState,
        const float* loc,
        void* outData,
        size_t dataSize) = 0;
    virtual void SampleLevel(
        SamplerState samplerState,
        const float* loc,
        float level,
        void* outData,
        size_t dataSize) = 0;
};

template<typename T>
struct Texture1D
{
    void GetDimensions(uint32_t* outWidth) { *outWidth = texture->GetDimensions().width; }
    void GetDimensions(uint32_t mipLevel, uint32_t* outWidth, uint32_t* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    void GetDimensions(float* outWidth) { *outWidth = texture->GetDimensions().width; }
    void GetDimensions(uint32_t mipLevel, float* outWidth, float* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int2& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T Sample(SamplerState samplerState, float loc) const
    {
        T out;
        texture->Sample(samplerState, &loc, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, float loc, float level) const
    {
        T out;
        texture->SampleLevel(samplerState, &loc, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct Texture2D
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int3& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T Sample(SamplerState samplerState, const float2& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float2& loc, float level) const
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct Texture3D
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outDepth)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outDepth,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight, float* outDepth)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outDepth,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int4& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T Sample(SamplerState samplerState, const float3& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float3& loc, float level) const
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct TextureCube
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Sample(SamplerState samplerState, const float3& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float3& loc, float level) const
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct Texture1DArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(float* outWidth, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outElements,
        float* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
        *outElements = dims.arrayElementCount;
    }

    T Load(const int3& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T Sample(SamplerState samplerState, const float2& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float2& loc, float level) const
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct Texture2DArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    void GetDimensions(uint32_t* outWidth, float* outHeight, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outElements,
        float* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int4& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T Sample(SamplerState samplerState, const float3& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float3& loc, float level) const
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

template<typename T>
struct TextureCubeArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    void GetDimensions(uint32_t* outWidth, float* outHeight, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outElements,
        float* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Sample(SamplerState samplerState, const float4& loc) const
    {
        T out;
        texture->Sample(samplerState, &loc.x, &out, sizeof(out));
        return out;
    }
    T SampleLevel(SamplerState samplerState, const float4& loc, float level) const
    {
        T out;
        texture->SampleLevel(samplerState, &loc.x, level, &out, sizeof(out));
        return out;
    }

    ITexture* texture;
};

/* !!!!!!!!!!!!!!!!!!!!!!!!!!! RWTexture !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

struct IRWTexture : ITexture
{
    /// Get the reference to the element at loc.
    virtual void* refAt(const uint32_t* loc) = 0;
};

template<typename T>
struct RWTexture1D
{
    void GetDimensions(uint32_t* outWidth) { *outWidth = texture->GetDimensions().width; }
    void GetDimensions(uint32_t mipLevel, uint32_t* outWidth, uint32_t* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    void GetDimensions(float* outWidth) { *outWidth = texture->GetDimensions().width; }
    void GetDimensions(uint32_t mipLevel, float* outWidth, float* outNumberOfLevels)
    {
        auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(int32_t loc) const
    {
        T out;
        texture->Load(&loc, &out, sizeof(out));
        return out;
    }
    T& operator[](uint32_t loc) { return *(T*)texture->refAt(&loc); }
    IRWTexture* texture;
};

template<typename T>
struct RWTexture2D
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int2& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T& operator[](const uint2& loc) { return *(T*)texture->refAt(&loc.x); }
    IRWTexture* texture;
};

template<typename T>
struct RWTexture3D
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outDepth)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outDepth,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight, float* outDepth)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outDepth,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outDepth = dims.depth;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int3& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T& operator[](const uint3& loc) { return *(T*)texture->refAt(&loc.x); }
    IRWTexture* texture;
};


template<typename T>
struct RWTexture1DArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outElements,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(int2 loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T& operator[](uint2 loc) { return *(T*)texture->refAt(&loc.x); }

    IRWTexture* texture;
};

template<typename T>
struct RWTexture2DArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outElements,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    T Load(const int3& loc) const
    {
        T out;
        texture->Load(&loc.x, &out, sizeof(out));
        return out;
    }
    T& operator[](const uint3& loc) { return *(T*)texture->refAt(&loc.x); }

    IRWTexture* texture;
};

// FeedbackTexture

struct FeedbackType
{
};
struct SAMPLER_FEEDBACK_MIN_MIP : FeedbackType
{
};
struct SAMPLER_FEEDBACK_MIP_REGION_USED : FeedbackType
{
};

struct IFeedbackTexture
{
    virtual TextureDimensions GetDimensions(int mipLevel = -1) = 0;

    // Note here we pass the optional clamp parameter as a pointer. Passing nullptr means no clamp.
    // This was preferred over having two function definitions, and having to differentiate their
    // names
    virtual void WriteSamplerFeedback(
        ITexture* tex,
        SamplerState samp,
        const float* location,
        const float* clamp = nullptr) = 0;
    virtual void WriteSamplerFeedbackBias(
        ITexture* tex,
        SamplerState samp,
        const float* location,
        float bias,
        const float* clamp = nullptr) = 0;
    virtual void WriteSamplerFeedbackGrad(
        ITexture* tex,
        SamplerState samp,
        const float* location,
        const float* ddx,
        const float* ddy,
        const float* clamp = nullptr) = 0;

    virtual void WriteSamplerFeedbackLevel(
        ITexture* tex,
        SamplerState samp,
        const float* location,
        float lod) = 0;
};

template<typename T>
struct FeedbackTexture2D
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight)
    {
        const auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    template<typename S>
    void WriteSamplerFeedback(Texture2D<S> tex, SamplerState samp, float2 location, float clamp)
    {
        texture->WriteSamplerFeedback(tex.texture, samp, &location.x, &clamp);
    }

    template<typename S>
    void WriteSamplerFeedbackBias(
        Texture2D<S> tex,
        SamplerState samp,
        float2 location,
        float bias,
        float clamp)
    {
        texture->WriteSamplerFeedbackBias(tex.texture, samp, &location.x, bias, &clamp);
    }

    template<typename S>
    void WriteSamplerFeedbackGrad(
        Texture2D<S> tex,
        SamplerState samp,
        float2 location,
        float2 ddx,
        float2 ddy,
        float clamp)
    {
        texture->WriteSamplerFeedbackGrad(tex.texture, samp, &location.x, &ddx.x, &ddy.x, &clamp);
    }

    // Level

    template<typename S>
    void WriteSamplerFeedbackLevel(Texture2D<S> tex, SamplerState samp, float2 location, float lod)
    {
        texture->WriteSamplerFeedbackLevel(tex.texture, samp, &location.x, lod);
    }

    // Without Clamp
    template<typename S>
    void WriteSamplerFeedback(Texture2D<S> tex, SamplerState samp, float2 location)
    {
        texture->WriteSamplerFeedback(tex.texture, samp, &location.x);
    }

    template<typename S>
    void WriteSamplerFeedbackBias(Texture2D<S> tex, SamplerState samp, float2 location, float bias)
    {
        texture->WriteSamplerFeedbackBias(tex.texture, samp, &location.x, bias);
    }

    template<typename S>
    void WriteSamplerFeedbackGrad(
        Texture2D<S> tex,
        SamplerState samp,
        float2 location,
        float2 ddx,
        float2 ddy)
    {
        texture->WriteSamplerFeedbackGrad(tex.texture, samp, &location.x, &ddx.x, &ddy.x);
    }

    IFeedbackTexture* texture;
};

template<typename T>
struct FeedbackTexture2DArray
{
    void GetDimensions(uint32_t* outWidth, uint32_t* outHeight, uint32_t* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        uint32_t* outWidth,
        uint32_t* outHeight,
        uint32_t* outElements,
        uint32_t* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }
    void GetDimensions(float* outWidth, float* outHeight, float* outElements)
    {
        auto dims = texture->GetDimensions();
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
    }
    void GetDimensions(
        uint32_t mipLevel,
        float* outWidth,
        float* outHeight,
        float* outElements,
        float* outNumberOfLevels)
    {
        const auto dims = texture->GetDimensions(mipLevel);
        *outWidth = dims.width;
        *outHeight = dims.height;
        *outElements = dims.arrayElementCount;
        *outNumberOfLevels = dims.numberOfLevels;
    }

    template<typename S>
    void WriteSamplerFeedback(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float clamp)
    {
        texture->WriteSamplerFeedback(texArray.texture, samp, &location.x, &clamp);
    }

    template<typename S>
    void WriteSamplerFeedbackBias(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float bias,
        float clamp)
    {
        texture->WriteSamplerFeedbackBias(texArray.texture, samp, &location.x, bias, &clamp);
    }

    template<typename S>
    void WriteSamplerFeedbackGrad(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float3 ddx,
        float3 ddy,
        float clamp)
    {
        texture
            ->WriteSamplerFeedbackGrad(texArray.texture, samp, &location.x, &ddx.x, &ddy.x, &clamp);
    }

    // Level
    template<typename S>
    void WriteSamplerFeedbackLevel(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float lod)
    {
        texture->WriteSamplerFeedbackLevel(texArray.texture, samp, &location.x, lod);
    }

    // Without Clamp

    template<typename S>
    void WriteSamplerFeedback(Texture2DArray<S> texArray, SamplerState samp, float3 location)
    {
        texture->WriteSamplerFeedback(texArray.texture, samp, &location.x);
    }

    template<typename S>
    void WriteSamplerFeedbackBias(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float bias)
    {
        texture->WriteSamplerFeedbackBias(texArray.texture, samp, &location.x, bias);
    }

    template<typename S>
    void WriteSamplerFeedbackGrad(
        Texture2DArray<S> texArray,
        SamplerState samp,
        float3 location,
        float3 ddx,
        float3 ddy)
    {
        texture->WriteSamplerFeedbackGrad(texArray.texture, samp, &location.x, &ddx.x, &ddy.x);
    }

    IFeedbackTexture* texture;
};

/* Varying input for Compute */

/* Used when running a single thread */
struct ComputeThreadVaryingInput
{
    uint3 groupID;
    uint3 groupThreadID;
};

struct ComputeVaryingInput
{
    uint3 startGroupID; ///< start groupID
    uint3 endGroupID;   ///< Non inclusive end groupID
};

// The uniformEntryPointParams and uniformState must be set to structures that match layout that the
// kernel expects. This can be determined via reflection for example.

typedef void (*ComputeThreadFunc)(
    ComputeThreadVaryingInput* varyingInput,
    void* uniformEntryPointParams,
    void* uniformState);
typedef void (*ComputeFunc)(
    ComputeVaryingInput* varyingInput,
    void* uniformEntryPointParams,
    void* uniformState);

#ifdef SLANG_PRELUDE_NAMESPACE
}
#endif

#endif


// Atomic helpers for the CPU target. Needed so kIROp_AtomicAdd and
// friends can lower to native atomic operations on the host - matching
// the semantics of HLSL `InterlockedAdd`, SPIR-V `OpAtomicIAdd`, etc.
// Uses compiler builtins so no <atomic> header is required.
//
// Contract for every `_slang_atomic_add_*` helper below (u32/i32/u64/
// i64): atomically add `val` to `*ptr` and return the PRIOR value (the
// value before the add), matching HLSL `InterlockedAdd` and GLSL
// `atomicAdd`. The 64-bit MSVC variants reinterpret the pointer as
// `volatile long long*` for `_InterlockedExchangeAdd64`, which is sound
// because `sizeof(long long) == 8` (asserted below).
//
// Compiler coverage:
//   - MSVC:            `_InterlockedExchangeAdd`
//   - GCC / Clang:     `__atomic_fetch_add`
//   - SNC / GHS etc.:  falls back to a non-atomic load/add/store.
//     These are console / embedded toolchains not known to ship
//     `__atomic_*` builtins; the fallback is racy under concurrency
//     but keeps the prelude buildable there. Platforms that need
//     real atomics on those compilers can add a bespoke branch.
// === MSVC implementations ===
//
// Use the `_Interlocked*` intrinsics. The 32-bit overload operates on
// `long`; the 64-bit overload operates on `long long`. The static
// asserts above pin the expected widths.
#if SLANG_VC
#include <intrin.h>
static_assert(
    sizeof(long) == 4,
    "_InterlockedExchangeAdd uses `long`; MSVC LLP64 requires sizeof(long)==4");
static_assert(
    sizeof(long long) == 8,
    "_InterlockedExchangeAdd64 uses `long long`; expected 8-byte width");
static inline uint32_t _slang_atomic_add_u32(uint32_t* ptr, uint32_t val)
{
    // Returns the PRIOR value, matching HLSL InterlockedAdd and GLSL atomicAdd.
    return static_cast<uint32_t>(
        _InterlockedExchangeAdd(reinterpret_cast<volatile long*>(ptr), static_cast<long>(val)));
}
static inline int32_t _slang_atomic_add_i32(int32_t* ptr, int32_t val)
{
    return static_cast<int32_t>(
        _InterlockedExchangeAdd(reinterpret_cast<volatile long*>(ptr), static_cast<long>(val)));
}
static inline uint64_t _slang_atomic_add_u64(uint64_t* ptr, uint64_t val)
{
    return static_cast<uint64_t>(_InterlockedExchangeAdd64(
        reinterpret_cast<volatile long long*>(ptr),
        static_cast<long long>(val)));
}
static inline int64_t _slang_atomic_add_i64(int64_t* ptr, int64_t val)
{
    return static_cast<int64_t>(_InterlockedExchangeAdd64(
        reinterpret_cast<volatile long long*>(ptr),
        static_cast<long long>(val)));
}
// === GCC / Clang implementations ===
//
// Use the `__atomic_fetch_add` built-in with relaxed ordering; the
// IR-side AtomicAdd emit ignores the memory-order operand and lets
// each toolchain pick its native default.
#elif SLANG_GCC || SLANG_CLANG
static inline uint32_t _slang_atomic_add_u32(uint32_t* ptr, uint32_t val)
{
    return __atomic_fetch_add(ptr, val, __ATOMIC_RELAXED);
}
static inline int32_t _slang_atomic_add_i32(int32_t* ptr, int32_t val)
{
    return __atomic_fetch_add(ptr, val, __ATOMIC_RELAXED);
}
static inline uint64_t _slang_atomic_add_u64(uint64_t* ptr, uint64_t val)
{
    return __atomic_fetch_add(ptr, val, __ATOMIC_RELAXED);
}
static inline int64_t _slang_atomic_add_i64(int64_t* ptr, int64_t val)
{
    return __atomic_fetch_add(ptr, val, __ATOMIC_RELAXED);
}
// === Non-atomic fallback implementations ===
//
// For compilers without a known atomic builtin (Sony SNC, Green Hills
// MULTI, etc.). Racy under concurrent invocation but keeps the
// prelude compilable; CPU-target coverage on these platforms is
// single-threaded in practice.
#else
static inline uint32_t _slang_atomic_add_u32(uint32_t* ptr, uint32_t val)
{
    uint32_t old = *ptr;
    *ptr = old + val;
    return old;
}
static inline int32_t _slang_atomic_add_i32(int32_t* ptr, int32_t val)
{
    int32_t old = *ptr;
    *ptr = old + val;
    return old;
}
static inline uint64_t _slang_atomic_add_u64(uint64_t* ptr, uint64_t val)
{
    uint64_t old = *ptr;
    *ptr = old + val;
    return old;
}
static inline int64_t _slang_atomic_add_i64(int64_t* ptr, int64_t val)
{
    int64_t old = *ptr;
    *ptr = old + val;
    return old;
}
#endif

// TODO(JS): Hack! Output C++ code from slang can copy uninitialized variables.
#if defined(_MSC_VER)
#pragma warning(disable : 4700)
#endif

#ifndef SLANG_UNROLL
#define SLANG_UNROLL
#endif

#endif

#ifdef SLANG_PRELUDE_NAMESPACE
using namespace SLANG_PRELUDE_NAMESPACE;
#endif


#line 1 "slang/rsqrt_zen2_f32.slang"
struct Slot_0
{
    uint32_t base_0;
    uint32_t pad0_0;
    uint32_t pad1_0;
    uint32_t pad2_0;
};


#line 1
struct GlobalParams_0
{
    StructuredBuffer<uint32_t> params_0;
    RWStructuredBuffer<float> s0_0;
    RWStructuredBuffer<uint32_t> s1_0;
    RWStructuredBuffer<uint32_t> s2_0;
    RWStructuredBuffer<float> dst_0;
    Slot_0* slot_0;
};


#line 12881 "hlsl.meta.slang"
struct KernelContext_0
{
    GlobalParams_0* globalParams_0;
};


#line 21 "slang/rsqrt_zen2_f32.slang"
static uint32_t pw_0(uint32_t k_0, KernelContext_0 * kernelContext_0)
{

#line 22
    return kernelContext_0->globalParams_0->params_0.Load(kernelContext_0->globalParams_0->slot_0->base_0 + k_0);
}

static void unravel4_0(uint32_t e_0, uint32_t w_0, uint32_t * i0_0, uint32_t * i1_0, uint32_t * i2_0, uint32_t * i3_0, KernelContext_0 * kernelContext_1)
{

#line 25
    uint32_t _S1 = pw_0(w_0, kernelContext_1);

#line 25
    uint32_t _S2 = pw_0(w_0 + 1U, kernelContext_1);

#line 25
    uint32_t _S3 = pw_0(w_0 + 2U, kernelContext_1);



    uint32_t _S4 = e_0 % _S1;

#line 29
    *i0_0 = _S4;
    uint32_t r_0 = e_0 / _S1;
    uint32_t _S5 = r_0 % _S2;

#line 31
    *i1_0 = _S5;
    uint32_t r_1 = r_0 / _S2;
    uint32_t _S6 = r_1 % _S3;

#line 33
    *i2_0 = _S6;
    uint32_t _S7 = r_1 / _S3;

#line 34
    *i3_0 = _S7;
    return;
}


#line 37
static uint32_t off4_0(uint32_t w_1, uint32_t i0_1, uint32_t i1_1, uint32_t i2_1, uint32_t i3_1, KernelContext_0 * kernelContext_2)
{

#line 37
    uint32_t _S8 = pw_0(w_1 + 8U, kernelContext_2);

#line 37
    uint32_t _S9 = pw_0(w_1 + 4U, kernelContext_2);
    uint32_t _S10 = _S8 + i0_1 * _S9;

#line 38
    uint32_t _S11 = pw_0(w_1 + 5U, kernelContext_2);

#line 38
    uint32_t _S12 = _S10 + i1_1 * _S11;

#line 38
    uint32_t _S13 = pw_0(w_1 + 6U, kernelContext_2);

#line 38
    uint32_t _S14 = _S12 + i2_1 * _S13;

#line 38
    uint32_t _S15 = pw_0(w_1 + 7U, kernelContext_2);

#line 38
    return _S14 + i3_1 * _S15;
}


#line 100
static uint32_t zen2_word_0(uint32_t w_2)
{

#line 101
    if(w_2 < 256U)
    {

#line 102
        if(w_2 < 128U)
        {

#line 103
            if(w_2 < 64U)
            {

#line 104
                if(w_2 < 32U)
                {

#line 105
                    if(w_2 < 16U)
                    {

#line 106
                        if(w_2 < 8U)
                        {

#line 107
                            if(w_2 < 4U)
                            {

#line 108
                                if(w_2 < 2U)
                                {

#line 109
                                    if(w_2 < 1U)
                                    {

#line 110
                                        return 1431633920U;
                                    }
                                    else
                                    {

#line 112
                                        return 1431633920U;
                                    }

#line 112
                                }
                                else
                                {
                                    if(w_2 < 3U)
                                    {

#line 116
                                        return 1431633920U;
                                    }
                                    else
                                    {

#line 118
                                        return 2863289685U;
                                    }

#line 118
                                }

#line 118
                            }
                            else
                            {

                                if(w_2 < 6U)
                                {

#line 123
                                    if(w_2 < 5U)
                                    {

#line 124
                                        return 1431983445U;
                                    }
                                    else
                                    {

#line 126
                                        return 1431633920U;
                                    }

#line 126
                                }
                                else
                                {
                                    if(w_2 < 7U)
                                    {

#line 130
                                        return 1431983445U;
                                    }
                                    else
                                    {

#line 132
                                        return 2863289685U;
                                    }

#line 132
                                }

#line 132
                            }

#line 132
                        }
                        else
                        {


                            if(w_2 < 12U)
                            {

#line 138
                                if(w_2 < 10U)
                                {

#line 139
                                    if(w_2 < 9U)
                                    {

#line 140
                                        return 1431633925U;
                                    }
                                    else
                                    {

#line 142
                                        return 1431635285U;
                                    }

#line 142
                                }
                                else
                                {
                                    if(w_2 < 11U)
                                    {

#line 146
                                        return 1431655765U;
                                    }
                                    else
                                    {

#line 148
                                        return 1431655765U;
                                    }

#line 148
                                }

#line 148
                            }
                            else
                            {

                                if(w_2 < 14U)
                                {

#line 153
                                    if(w_2 < 13U)
                                    {

#line 154
                                        return 1431639381U;
                                    }
                                    else
                                    {

#line 156
                                        return 1431635285U;
                                    }

#line 156
                                }
                                else
                                {
                                    if(w_2 < 15U)
                                    {

#line 160
                                        return 357892117U;
                                    }
                                    else
                                    {

#line 162
                                        return 1433032021U;
                                    }

#line 162
                                }

#line 162
                            }

#line 162
                        }

#line 162
                    }
                    else
                    {



                        if(w_2 < 24U)
                        {

#line 169
                            if(w_2 < 20U)
                            {

#line 170
                                if(w_2 < 18U)
                                {

#line 171
                                    if(w_2 < 17U)
                                    {

#line 172
                                        return 2857724501U;
                                    }
                                    else
                                    {

#line 174
                                        return 1431655685U;
                                    }

#line 174
                                }
                                else
                                {
                                    if(w_2 < 19U)
                                    {

#line 178
                                        return 2857724501U;
                                    }
                                    else
                                    {

#line 180
                                        return 1426412801U;
                                    }

#line 180
                                }

#line 180
                            }
                            else
                            {

                                if(w_2 < 22U)
                                {

#line 185
                                    if(w_2 < 21U)
                                    {

#line 186
                                        return 1448432981U;
                                    }
                                    else
                                    {

#line 188
                                        return 1515543125U;
                                    }

#line 188
                                }
                                else
                                {
                                    if(w_2 < 23U)
                                    {

#line 192
                                        return 1783982677U;
                                    }
                                    else
                                    {

#line 194
                                        return 2857740885U;
                                    }

#line 194
                                }

#line 194
                            }

#line 194
                        }
                        else
                        {


                            if(w_2 < 28U)
                            {

#line 200
                                if(w_2 < 26U)
                                {

#line 201
                                    if(w_2 < 25U)
                                    {

#line 202
                                        return 1783982677U;
                                    }
                                    else
                                    {

#line 204
                                        return 1515547221U;
                                    }

#line 204
                                }
                                else
                                {
                                    if(w_2 < 27U)
                                    {

#line 208
                                        return 1448434261U;
                                    }
                                    else
                                    {

#line 210
                                        return 1427461461U;
                                    }

#line 210
                                }

#line 210
                            }
                            else
                            {

                                if(w_2 < 30U)
                                {

#line 215
                                    if(w_2 < 29U)
                                    {

#line 216
                                        return 1426150661U;
                                    }
                                    else
                                    {

#line 218
                                        return 1448434261U;
                                    }

#line 218
                                }
                                else
                                {
                                    if(w_2 < 31U)
                                    {

#line 222
                                        return 1426412885U;
                                    }
                                    else
                                    {

#line 224
                                        return 1448438357U;
                                    }

#line 224
                                }

#line 224
                            }

#line 224
                        }

#line 224
                    }

#line 224
                }
                else
                {

#line 231
                    if(w_2 < 48U)
                    {

#line 232
                        if(w_2 < 40U)
                        {

#line 233
                            if(w_2 < 36U)
                            {

#line 234
                                if(w_2 < 34U)
                                {

#line 235
                                    if(w_2 < 33U)
                                    {

#line 236
                                        return 1700156757U;
                                    }
                                    else
                                    {

#line 238
                                        return 2775209621U;
                                    }

#line 238
                                }
                                else
                                {
                                    if(w_2 < 35U)
                                    {

#line 242
                                        return 1364530497U;
                                    }
                                    else
                                    {

#line 244
                                        return 1431655749U;
                                    }

#line 244
                                }

#line 244
                            }
                            else
                            {

                                if(w_2 < 38U)
                                {

#line 249
                                    if(w_2 < 37U)
                                    {

#line 250
                                        return 1431721301U;
                                    }
                                    else
                                    {

#line 252
                                        return 1700156821U;
                                    }

#line 252
                                }
                                else
                                {
                                    if(w_2 < 39U)
                                    {

#line 256
                                        return 1431721365U;
                                    }
                                    else
                                    {

#line 258
                                        return 1431721365U;
                                    }

#line 258
                                }

#line 258
                            }

#line 258
                        }
                        else
                        {


                            if(w_2 < 44U)
                            {

#line 264
                                if(w_2 < 42U)
                                {

#line 265
                                    if(w_2 < 41U)
                                    {

#line 266
                                        return 1364546901U;
                                    }
                                    else
                                    {

#line 268
                                        return 1347753285U;
                                    }

#line 268
                                }
                                else
                                {
                                    if(w_2 < 43U)
                                    {

#line 272
                                        return 2775210646U;
                                    }
                                    else
                                    {

#line 274
                                        return 1431721365U;
                                    }

#line 274
                                }

#line 274
                            }
                            else
                            {

                                if(w_2 < 46U)
                                {

#line 279
                                    if(w_2 < 45U)
                                    {

#line 280
                                        return 1347753301U;
                                    }
                                    else
                                    {

#line 282
                                        return 1431983766U;
                                    }

#line 282
                                }
                                else
                                {
                                    if(w_2 < 47U)
                                    {

#line 286
                                        return 1347769685U;
                                    }
                                    else
                                    {

#line 288
                                        return 1431721621U;
                                    }

#line 288
                                }

#line 288
                            }

#line 288
                        }

#line 288
                    }
                    else
                    {



                        if(w_2 < 56U)
                        {

#line 295
                            if(w_2 < 52U)
                            {

#line 296
                                if(w_2 < 50U)
                                {

#line 297
                                    if(w_2 < 49U)
                                    {

#line 298
                                        return 1701143877U;
                                    }
                                    else
                                    {

#line 300
                                        return 2779096469U;
                                    }

#line 300
                                }
                                else
                                {
                                    if(w_2 < 51U)
                                    {

#line 304
                                        return 1364283712U;
                                    }
                                    else
                                    {

#line 306
                                        return 1431655745U;
                                    }

#line 306
                                }

#line 306
                            }
                            else
                            {

                                if(w_2 < 54U)
                                {

#line 311
                                    if(w_2 < 53U)
                                    {

#line 312
                                        return 1431655749U;
                                    }
                                    else
                                    {

#line 314
                                        return 1431655749U;
                                    }

#line 314
                                }
                                else
                                {
                                    if(w_2 < 55U)
                                    {

#line 318
                                        return 1431655749U;
                                    }
                                    else
                                    {

#line 320
                                        return 1431655749U;
                                    }

#line 320
                                }

#line 320
                            }

#line 320
                        }
                        else
                        {


                            if(w_2 < 60U)
                            {

#line 326
                                if(w_2 < 58U)
                                {

#line 327
                                    if(w_2 < 57U)
                                    {

#line 328
                                        return 1431655749U;
                                    }
                                    else
                                    {

#line 330
                                        return 1364283713U;
                                    }

#line 330
                                }
                                else
                                {
                                    if(w_2 < 59U)
                                    {

#line 334
                                        return 2779096470U;
                                    }
                                    else
                                    {

#line 336
                                        return 1701143893U;
                                    }

#line 336
                                }

#line 336
                            }
                            else
                            {

                                if(w_2 < 62U)
                                {

#line 341
                                    if(w_2 < 61U)
                                    {

#line 342
                                        return 1364546885U;
                                    }
                                    else
                                    {

#line 344
                                        return 1705354902U;
                                    }

#line 344
                                }
                                else
                                {
                                    if(w_2 < 63U)
                                    {

#line 348
                                        return 1431659861U;
                                    }
                                    else
                                    {

#line 350
                                        return 1705354902U;
                                    }

#line 350
                                }

#line 350
                            }

#line 350
                        }

#line 350
                    }

#line 350
                }

#line 350
            }
            else
            {

#line 358
                if(w_2 < 96U)
                {

#line 359
                    if(w_2 < 80U)
                    {

#line 360
                        if(w_2 < 72U)
                        {

#line 361
                            if(w_2 < 68U)
                            {

#line 362
                                if(w_2 < 66U)
                                {

#line 363
                                    if(w_2 < 65U)
                                    {

#line 364
                                        return 1363477780U;
                                    }
                                    else
                                    {

#line 366
                                        return 1700091221U;
                                    }

#line 366
                                }
                                else
                                {
                                    if(w_2 < 67U)
                                    {

#line 370
                                        return 2778094169U;
                                    }
                                    else
                                    {

#line 372
                                        return 1363481876U;
                                    }

#line 372
                                }

#line 372
                            }
                            else
                            {

                                if(w_2 < 70U)
                                {

#line 377
                                    if(w_2 < 69U)
                                    {

#line 378
                                        return 1431639381U;
                                    }
                                    else
                                    {

#line 380
                                        return 1700091477U;
                                    }

#line 380
                                }
                                else
                                {
                                    if(w_2 < 71U)
                                    {

#line 384
                                        return 1704351321U;
                                    }
                                    else
                                    {

#line 386
                                        return 1704352345U;
                                    }

#line 386
                                }

#line 386
                            }

#line 386
                        }
                        else
                        {


                            if(w_2 < 76U)
                            {

#line 392
                                if(w_2 < 74U)
                                {

#line 393
                                    if(w_2 < 73U)
                                    {

#line 394
                                        return 1704352345U;
                                    }
                                    else
                                    {

#line 396
                                        return 1704352345U;
                                    }

#line 396
                                }
                                else
                                {
                                    if(w_2 < 75U)
                                    {

#line 400
                                        return 1704351321U;
                                    }
                                    else
                                    {

#line 402
                                        return 1431656025U;
                                    }

#line 402
                                }

#line 402
                            }
                            else
                            {

                                if(w_2 < 78U)
                                {

#line 407
                                    if(w_2 < 77U)
                                    {

#line 408
                                        return 1431656025U;
                                    }
                                    else
                                    {

#line 410
                                        return 1363481941U;
                                    }

#line 410
                                }
                                else
                                {
                                    if(w_2 < 79U)
                                    {

#line 414
                                        return 1704352361U;
                                    }
                                    else
                                    {

#line 416
                                        return 1431656025U;
                                    }

#line 416
                                }

#line 416
                            }

#line 416
                        }

#line 416
                    }
                    else
                    {



                        if(w_2 < 88U)
                        {

#line 423
                            if(w_2 < 84U)
                            {

#line 424
                                if(w_2 < 82U)
                                {

#line 425
                                    if(w_2 < 81U)
                                    {

#line 426
                                        return 1431639121U;
                                    }
                                    else
                                    {

#line 428
                                        return 1772509541U;
                                    }

#line 428
                                }
                                else
                                {
                                    if(w_2 < 83U)
                                    {

#line 432
                                        return 1499813205U;
                                    }
                                    else
                                    {

#line 434
                                        return 1772509541U;
                                    }

#line 434
                                }

#line 434
                            }
                            else
                            {

                                if(w_2 < 86U)
                                {

#line 439
                                    if(w_2 < 85U)
                                    {

#line 440
                                        return 1431655765U;
                                    }
                                    else
                                    {

#line 442
                                        return 1504074085U;
                                    }

#line 442
                                }
                                else
                                {
                                    if(w_2 < 87U)
                                    {

#line 446
                                        return 357897297U;
                                    }
                                    else
                                    {

#line 448
                                        return 1432704341U;
                                    }

#line 448
                                }

#line 448
                            }

#line 448
                        }
                        else
                        {


                            if(w_2 < 92U)
                            {

#line 454
                                if(w_2 < 90U)
                                {

#line 455
                                    if(w_2 < 89U)
                                    {

#line 456
                                        return 1499879781U;
                                    }
                                    else
                                    {

#line 458
                                        return 1772509606U;
                                    }

#line 458
                                }
                                else
                                {
                                    if(w_2 < 91U)
                                    {

#line 462
                                        return 341120081U;
                                    }
                                    else
                                    {

#line 464
                                        return 357897557U;
                                    }

#line 464
                                }

#line 464
                            }
                            else
                            {

                                if(w_2 < 94U)
                                {

#line 469
                                    if(w_2 < 93U)
                                    {

#line 470
                                        return 1431655765U;
                                    }
                                    else
                                    {

#line 472
                                        return 1431655765U;
                                    }

#line 472
                                }
                                else
                                {
                                    if(w_2 < 95U)
                                    {

#line 476
                                        return 1431655765U;
                                    }
                                    else
                                    {

#line 478
                                        return 1431655765U;
                                    }

#line 478
                                }

#line 478
                            }

#line 478
                        }

#line 478
                    }

#line 478
                }
                else
                {

#line 485
                    if(w_2 < 112U)
                    {

#line 486
                        if(w_2 < 104U)
                        {

#line 487
                            if(w_2 < 100U)
                            {

#line 488
                                if(w_2 < 98U)
                                {

#line 489
                                    if(w_2 < 97U)
                                    {

#line 490
                                        return 1721394777U;
                                    }
                                    else
                                    {

#line 492
                                        return 1704617561U;
                                    }

#line 492
                                }
                                else
                                {
                                    if(w_2 < 99U)
                                    {

#line 496
                                        return 1704355161U;
                                    }
                                    else
                                    {

#line 498
                                        return 1431721301U;
                                    }

#line 498
                                }

#line 498
                            }
                            else
                            {

                                if(w_2 < 102U)
                                {

#line 503
                                    if(w_2 < 101U)
                                    {

#line 504
                                        return 1364546837U;
                                    }
                                    else
                                    {

#line 506
                                        return 1721394777U;
                                    }

#line 506
                                }
                                else
                                {
                                    if(w_2 < 103U)
                                    {

#line 510
                                        return 1435919705U;
                                    }
                                    else
                                    {

#line 512
                                        return 1364546837U;
                                    }

#line 512
                                }

#line 512
                            }

#line 512
                        }
                        else
                        {


                            if(w_2 < 108U)
                            {

#line 518
                                if(w_2 < 106U)
                                {

#line 519
                                    if(w_2 < 105U)
                                    {

#line 520
                                        return 1704617561U;
                                    }
                                    else
                                    {

#line 522
                                        return 1431721301U;
                                    }

#line 522
                                }
                                else
                                {
                                    if(w_2 < 107U)
                                    {

#line 526
                                        return 1704617561U;
                                    }
                                    else
                                    {

#line 528
                                        return 1364612437U;
                                    }

#line 528
                                }

#line 528
                            }
                            else
                            {

                                if(w_2 < 110U)
                                {

#line 533
                                    if(w_2 < 109U)
                                    {

#line 534
                                        return 1436182105U;
                                    }
                                    else
                                    {

#line 536
                                        return 290805013U;
                                    }

#line 536
                                }
                                else
                                {
                                    if(w_2 < 111U)
                                    {

#line 540
                                        return 1431725401U;
                                    }
                                    else
                                    {

#line 542
                                        return 1704617577U;
                                    }

#line 542
                                }

#line 542
                            }

#line 542
                        }

#line 542
                    }
                    else
                    {



                        if(w_2 < 120U)
                        {

#line 549
                            if(w_2 < 116U)
                            {

#line 550
                                if(w_2 < 114U)
                                {

#line 551
                                    if(w_2 < 113U)
                                    {

#line 552
                                        return 1448434005U;
                                    }
                                    else
                                    {

#line 554
                                        return 2523290009U;
                                    }

#line 554
                                }
                                else
                                {
                                    if(w_2 < 115U)
                                    {

#line 558
                                        return 2594597273U;
                                    }
                                    else
                                    {

#line 560
                                        return 1163203924U;
                                    }

#line 560
                                }

#line 560
                            }
                            else
                            {

                                if(w_2 < 118U)
                                {

#line 565
                                    if(w_2 < 117U)
                                    {

#line 566
                                        return 1163220309U;
                                    }
                                    else
                                    {

#line 568
                                        return 1431655765U;
                                    }

#line 568
                                }
                                else
                                {
                                    if(w_2 < 119U)
                                    {

#line 572
                                        return 1431656789U;
                                    }
                                    else
                                    {

#line 574
                                        return 1448434005U;
                                    }

#line 574
                                }

#line 574
                            }

#line 574
                        }
                        else
                        {


                            if(w_2 < 124U)
                            {

#line 580
                                if(w_2 < 122U)
                                {

#line 581
                                    if(w_2 < 121U)
                                    {

#line 582
                                        return 1431656789U;
                                    }
                                    else
                                    {

#line 584
                                        return 1431656789U;
                                    }

#line 584
                                }
                                else
                                {
                                    if(w_2 < 123U)
                                    {

#line 588
                                        return 1163220309U;
                                    }
                                    else
                                    {

#line 590
                                        return 1163203925U;
                                    }

#line 590
                                }

#line 590
                            }
                            else
                            {

                                if(w_2 < 126U)
                                {

#line 595
                                    if(w_2 < 125U)
                                    {

#line 596
                                        return 1158747476U;
                                    }
                                    else
                                    {

#line 598
                                        return 1449552281U;
                                    }

#line 598
                                }
                                else
                                {
                                    if(w_2 < 127U)
                                    {

#line 602
                                        return 1449482649U;
                                    }
                                    else
                                    {

#line 604
                                        return 1163221333U;
                                    }

#line 604
                                }

#line 604
                            }

#line 604
                        }

#line 604
                    }

#line 604
                }

#line 604
            }

#line 604
        }
        else
        {

#line 613
            if(w_2 < 192U)
            {

#line 614
                if(w_2 < 160U)
                {

#line 615
                    if(w_2 < 144U)
                    {

#line 616
                        if(w_2 < 136U)
                        {

#line 617
                            if(w_2 < 132U)
                            {

#line 618
                                if(w_2 < 130U)
                                {

#line 619
                                    if(w_2 < 129U)
                                    {

#line 620
                                        return 2863031910U;
                                    }
                                    else
                                    {

#line 622
                                        return 1717982549U;
                                    }

#line 622
                                }
                                else
                                {
                                    if(w_2 < 131U)
                                    {

#line 626
                                        return 1431655761U;
                                    }
                                    else
                                    {

#line 628
                                        return 2791728742U;
                                    }

#line 628
                                }

#line 628
                            }
                            else
                            {

                                if(w_2 < 134U)
                                {

#line 633
                                    if(w_2 < 133U)
                                    {

#line 634
                                        return 1716868437U;
                                    }
                                    else
                                    {

#line 636
                                        return 2863031910U;
                                    }

#line 636
                                }
                                else
                                {
                                    if(w_2 < 135U)
                                    {

#line 640
                                        return 1716868437U;
                                    }
                                    else
                                    {

#line 642
                                        return 2858837606U;
                                    }

#line 642
                                }

#line 642
                            }

#line 642
                        }
                        else
                        {


                            if(w_2 < 140U)
                            {

#line 648
                                if(w_2 < 138U)
                                {

#line 649
                                    if(w_2 < 137U)
                                    {

#line 650
                                        return 1700091221U;
                                    }
                                    else
                                    {

#line 652
                                        return 1717986918U;
                                    }

#line 652
                                }
                                else
                                {
                                    if(w_2 < 139U)
                                    {

#line 656
                                        return 1431655761U;
                                    }
                                    else
                                    {

#line 658
                                        return 1717917013U;
                                    }

#line 658
                                }

#line 658
                            }
                            else
                            {

                                if(w_2 < 142U)
                                {

#line 663
                                    if(w_2 < 141U)
                                    {

#line 664
                                        return 1717986918U;
                                    }
                                    else
                                    {

#line 666
                                        return 1431655697U;
                                    }

#line 666
                                }
                                else
                                {
                                    if(w_2 < 143U)
                                    {

#line 670
                                        return 1431655765U;
                                    }
                                    else
                                    {

#line 672
                                        return 1717986901U;
                                    }

#line 672
                                }

#line 672
                            }

#line 672
                        }

#line 672
                    }
                    else
                    {



                        if(w_2 < 152U)
                        {

#line 679
                            if(w_2 < 148U)
                            {

#line 680
                                if(w_2 < 146U)
                                {

#line 681
                                    if(w_2 < 145U)
                                    {

#line 682
                                        return 1431655765U;
                                    }
                                    else
                                    {

#line 684
                                        return 1431655765U;
                                    }

#line 684
                                }
                                else
                                {
                                    if(w_2 < 147U)
                                    {

#line 688
                                        return 2576980373U;
                                    }
                                    else
                                    {

#line 690
                                        return 2576980377U;
                                    }

#line 690
                                }

#line 690
                            }
                            else
                            {

                                if(w_2 < 150U)
                                {

#line 695
                                    if(w_2 < 149U)
                                    {

#line 696
                                        return 2576980377U;
                                    }
                                    else
                                    {

#line 698
                                        return 2576980377U;
                                    }

#line 698
                                }
                                else
                                {
                                    if(w_2 < 151U)
                                    {

#line 702
                                        return 2576980377U;
                                    }
                                    else
                                    {

#line 704
                                        return 2576980377U;
                                    }

#line 704
                                }

#line 704
                            }

#line 704
                        }
                        else
                        {


                            if(w_2 < 156U)
                            {

#line 710
                                if(w_2 < 154U)
                                {

#line 711
                                    if(w_2 < 153U)
                                    {

#line 712
                                        return 2576980377U;
                                    }
                                    else
                                    {

#line 714
                                        return 2576980377U;
                                    }

#line 714
                                }
                                else
                                {
                                    if(w_2 < 155U)
                                    {

#line 718
                                        return 1431673241U;
                                    }
                                    else
                                    {

#line 720
                                        return 1431655765U;
                                    }

#line 720
                                }

#line 720
                            }
                            else
                            {

                                if(w_2 < 158U)
                                {

#line 725
                                    if(w_2 < 157U)
                                    {

#line 726
                                        return 1431655765U;
                                    }
                                    else
                                    {

#line 728
                                        return 2577050282U;
                                    }

#line 728
                                }
                                else
                                {
                                    if(w_2 < 159U)
                                    {

#line 732
                                        return 2576980377U;
                                    }
                                    else
                                    {

#line 734
                                        return 1431935385U;
                                    }

#line 734
                                }

#line 734
                            }

#line 734
                        }

#line 734
                    }

#line 734
                }
                else
                {

#line 741
                    if(w_2 < 176U)
                    {

#line 742
                        if(w_2 < 168U)
                        {

#line 743
                            if(w_2 < 164U)
                            {

#line 744
                                if(w_2 < 162U)
                                {

#line 745
                                    if(w_2 < 161U)
                                    {

#line 746
                                        return 1717982549U;
                                    }
                                    else
                                    {

#line 748
                                        return 1431655765U;
                                    }

#line 748
                                }
                                else
                                {
                                    if(w_2 < 163U)
                                    {

#line 752
                                        return 1789565337U;
                                    }
                                    else
                                    {

#line 754
                                        return 1717982549U;
                                    }

#line 754
                                }

#line 754
                            }
                            else
                            {

                                if(w_2 < 166U)
                                {

#line 759
                                    if(w_2 < 165U)
                                    {

#line 760
                                        return 1431651413U;
                                    }
                                    else
                                    {

#line 762
                                        return 1717983641U;
                                    }

#line 762
                                }
                                else
                                {
                                    if(w_2 < 167U)
                                    {

#line 766
                                        return 1431721301U;
                                    }
                                    else
                                    {

#line 768
                                        return 1718262169U;
                                    }

#line 768
                                }

#line 768
                            }

#line 768
                        }
                        else
                        {


                            if(w_2 < 172U)
                            {

#line 774
                                if(w_2 < 170U)
                                {

#line 775
                                    if(w_2 < 169U)
                                    {

#line 776
                                        return 1432769877U;
                                    }
                                    else
                                    {

#line 778
                                        return 1718262169U;
                                    }

#line 778
                                }
                                else
                                {
                                    if(w_2 < 171U)
                                    {

#line 782
                                        return 1431655765U;
                                    }
                                    else
                                    {

#line 784
                                        return 1717983641U;
                                    }

#line 784
                                }

#line 784
                            }
                            else
                            {

                                if(w_2 < 174U)
                                {

#line 789
                                    if(w_2 < 173U)
                                    {

#line 790
                                        return 1431651413U;
                                    }
                                    else
                                    {

#line 792
                                        return 1449547093U;
                                    }

#line 792
                                }
                                else
                                {
                                    if(w_2 < 175U)
                                    {

#line 796
                                        return 1718262169U;
                                    }
                                    else
                                    {

#line 798
                                        return 1431651669U;
                                    }

#line 798
                                }

#line 798
                            }

#line 798
                        }

#line 798
                    }
                    else
                    {



                        if(w_2 < 184U)
                        {

#line 805
                            if(w_2 < 180U)
                            {

#line 806
                                if(w_2 < 178U)
                                {

#line 807
                                    if(w_2 < 177U)
                                    {

#line 808
                                        return 1414878481U;
                                    }
                                    else
                                    {

#line 810
                                        return 1700092245U;
                                    }

#line 810
                                }
                                else
                                {
                                    if(w_2 < 179U)
                                    {

#line 814
                                        return 1704565078U;
                                    }
                                    else
                                    {

#line 816
                                        return 2845420134U;
                                    }

#line 816
                                }

#line 816
                            }
                            else
                            {

                                if(w_2 < 182U)
                                {

#line 821
                                    if(w_2 < 181U)
                                    {

#line 822
                                        return 1414878481U;
                                    }
                                    else
                                    {

#line 824
                                        return 1431655765U;
                                    }

#line 824
                                }
                                else
                                {
                                    if(w_2 < 183U)
                                    {

#line 828
                                        return 1700092245U;
                                    }
                                    else
                                    {

#line 830
                                        return 1700370773U;
                                    }

#line 830
                                }

#line 830
                            }

#line 830
                        }
                        else
                        {


                            if(w_2 < 188U)
                            {

#line 836
                                if(w_2 < 186U)
                                {

#line 837
                                    if(w_2 < 185U)
                                    {

#line 838
                                        return 1700370774U;
                                    }
                                    else
                                    {

#line 840
                                        return 1704565094U;
                                    }

#line 840
                                }
                                else
                                {
                                    if(w_2 < 187U)
                                    {

#line 844
                                        return 1704565350U;
                                    }
                                    else
                                    {

#line 846
                                        return 1771674214U;
                                    }

#line 846
                                }

#line 846
                            }
                            else
                            {

                                if(w_2 < 190U)
                                {

#line 851
                                    if(w_2 < 189U)
                                    {

#line 852
                                        return 1771674214U;
                                    }
                                    else
                                    {

#line 854
                                        return 1704565350U;
                                    }

#line 854
                                }
                                else
                                {
                                    if(w_2 < 191U)
                                    {

#line 858
                                        return 1704565350U;
                                    }
                                    else
                                    {

#line 860
                                        return 1700370790U;
                                    }

#line 860
                                }

#line 860
                            }

#line 860
                        }

#line 860
                    }

#line 860
                }

#line 860
            }
            else
            {

#line 868
                if(w_2 < 224U)
                {

#line 869
                    if(w_2 < 208U)
                    {

#line 870
                        if(w_2 < 200U)
                        {

#line 871
                            if(w_2 < 196U)
                            {

#line 872
                                if(w_2 < 194U)
                                {

#line 873
                                    if(w_2 < 193U)
                                    {

#line 874
                                        return 1431917909U;
                                    }
                                    else
                                    {

#line 876
                                        return 1431655749U;
                                    }

#line 876
                                }
                                else
                                {
                                    if(w_2 < 195U)
                                    {

#line 880
                                        return 1163203908U;
                                    }
                                    else
                                    {

#line 882
                                        return 2578081433U;
                                    }

#line 882
                                }

#line 882
                            }
                            else
                            {

                                if(w_2 < 198U)
                                {

#line 887
                                    if(w_2 < 197U)
                                    {

#line 888
                                        return 1503290969U;
                                    }
                                    else
                                    {

#line 890
                                        return 1436112469U;
                                    }

#line 890
                                }
                                else
                                {
                                    if(w_2 < 199U)
                                    {

#line 894
                                        return 1431917909U;
                                    }
                                    else
                                    {

#line 896
                                        return 1163203908U;
                                    }

#line 896
                                }

#line 896
                            }

#line 896
                        }
                        else
                        {


                            if(w_2 < 204U)
                            {

#line 902
                                if(w_2 < 202U)
                                {

#line 903
                                    if(w_2 < 201U)
                                    {

#line 904
                                        return 2577032793U;
                                    }
                                    else
                                    {

#line 906
                                        return 1436112469U;
                                    }

#line 906
                                }
                                else
                                {
                                    if(w_2 < 203U)
                                    {

#line 910
                                        return 1163220293U;
                                    }
                                    else
                                    {

#line 912
                                        return 2577032857U;
                                    }

#line 912
                                }

#line 912
                            }
                            else
                            {

                                if(w_2 < 206U)
                                {

#line 917
                                    if(w_2 < 205U)
                                    {

#line 918
                                        return 1436112469U;
                                    }
                                    else
                                    {

#line 920
                                        return 1163203909U;
                                    }

#line 920
                                }
                                else
                                {
                                    if(w_2 < 207U)
                                    {

#line 924
                                        return 1503225433U;
                                    }
                                    else
                                    {

#line 926
                                        return 1431655765U;
                                    }

#line 926
                                }

#line 926
                            }

#line 926
                        }

#line 926
                    }
                    else
                    {



                        if(w_2 < 216U)
                        {

#line 933
                            if(w_2 < 212U)
                            {

#line 934
                                if(w_2 < 210U)
                                {

#line 935
                                    if(w_2 < 209U)
                                    {

#line 936
                                        return 1771464089U;
                                    }
                                    else
                                    {

#line 938
                                        return 1700091221U;
                                    }

#line 938
                                }
                                else
                                {
                                    if(w_2 < 211U)
                                    {

#line 942
                                        return 2845205913U;
                                    }
                                    else
                                    {

#line 944
                                        return 1700091221U;
                                    }

#line 944
                                }

#line 944
                            }
                            else
                            {

                                if(w_2 < 214U)
                                {

#line 949
                                    if(w_2 < 213U)
                                    {

#line 950
                                        return 1771464089U;
                                    }
                                    else
                                    {

#line 952
                                        return 1431393365U;
                                    }

#line 952
                                }
                                else
                                {
                                    if(w_2 < 215U)
                                    {

#line 956
                                        return 1771464089U;
                                    }
                                    else
                                    {

#line 958
                                        return 1431393365U;
                                    }

#line 958
                                }

#line 958
                            }

#line 958
                        }
                        else
                        {


                            if(w_2 < 220U)
                            {

#line 964
                                if(w_2 < 218U)
                                {

#line 965
                                    if(w_2 < 217U)
                                    {

#line 966
                                        return 1704289625U;
                                    }
                                    else
                                    {

#line 968
                                        return 1771464090U;
                                    }

#line 968
                                }
                                else
                                {
                                    if(w_2 < 219U)
                                    {

#line 972
                                        return 1431393365U;
                                    }
                                    else
                                    {

#line 974
                                        return 1704289625U;
                                    }

#line 974
                                }

#line 974
                            }
                            else
                            {

                                if(w_2 < 222U)
                                {

#line 979
                                    if(w_2 < 221U)
                                    {

#line 980
                                        return 1771464090U;
                                    }
                                    else
                                    {

#line 982
                                        return 1431393365U;
                                    }

#line 982
                                }
                                else
                                {
                                    if(w_2 < 223U)
                                    {

#line 986
                                        return 1700091221U;
                                    }
                                    else
                                    {

#line 988
                                        return 1704289689U;
                                    }

#line 988
                                }

#line 988
                            }

#line 988
                        }

#line 988
                    }

#line 988
                }
                else
                {

#line 995
                    if(w_2 < 240U)
                    {

#line 996
                        if(w_2 < 232U)
                        {

#line 997
                            if(w_2 < 228U)
                            {

#line 998
                                if(w_2 < 226U)
                                {

#line 999
                                    if(w_2 < 225U)
                                    {

#line 1000
                                        return 340071749U;
                                    }
                                    else
                                    {

#line 1002
                                        return 1414878549U;
                                    }

#line 1002
                                }
                                else
                                {
                                    if(w_2 < 227U)
                                    {

#line 1006
                                        return 1431918165U;
                                    }
                                    else
                                    {

#line 1008
                                        return 1700357721U;
                                    }

#line 1008
                                }

#line 1008
                            }
                            else
                            {

                                if(w_2 < 230U)
                                {

#line 1013
                                    if(w_2 < 229U)
                                    {

#line 1014
                                        return 1704552089U;
                                    }
                                    else
                                    {

#line 1016
                                        return 1704617626U;
                                    }

#line 1016
                                }
                                else
                                {
                                    if(w_2 < 231U)
                                    {

#line 1020
                                        return 1771727514U;
                                    }
                                    else
                                    {

#line 1022
                                        return 341120341U;
                                    }

#line 1022
                                }

#line 1022
                            }

#line 1022
                        }
                        else
                        {


                            if(w_2 < 236U)
                            {

#line 1028
                                if(w_2 < 234U)
                                {

#line 1029
                                    if(w_2 < 233U)
                                    {

#line 1030
                                        return 1414862165U;
                                    }
                                    else
                                    {

#line 1032
                                        return 1414878549U;
                                    }

#line 1032
                                }
                                else
                                {
                                    if(w_2 < 235U)
                                    {

#line 1036
                                        return 1414878549U;
                                    }
                                    else
                                    {

#line 1038
                                        return 1431655765U;
                                    }

#line 1038
                                }

#line 1038
                            }
                            else
                            {

                                if(w_2 < 238U)
                                {

#line 1043
                                    if(w_2 < 237U)
                                    {

#line 1044
                                        return 1431656021U;
                                    }
                                    else
                                    {

#line 1046
                                        return 1414878549U;
                                    }

#line 1046
                                }
                                else
                                {
                                    if(w_2 < 239U)
                                    {

#line 1050
                                        return 1414878549U;
                                    }
                                    else
                                    {

#line 1052
                                        return 1414878549U;
                                    }

#line 1052
                                }

#line 1052
                            }

#line 1052
                        }

#line 1052
                    }
                    else
                    {



                        if(w_2 < 248U)
                        {

#line 1059
                            if(w_2 < 244U)
                            {

#line 1060
                                if(w_2 < 242U)
                                {

#line 1061
                                    if(w_2 < 241U)
                                    {

#line 1062
                                        return 1499814245U;
                                    }
                                    else
                                    {

#line 1064
                                        return 1498765669U;
                                    }

#line 1064
                                }
                                else
                                {
                                    if(w_2 < 243U)
                                    {

#line 1068
                                        return 1431656789U;
                                    }
                                    else
                                    {

#line 1070
                                        return 1431655765U;
                                    }

#line 1070
                                }

#line 1070
                            }
                            else
                            {

                                if(w_2 < 246U)
                                {

#line 1075
                                    if(w_2 < 245U)
                                    {

#line 1076
                                        return 1427395924U;
                                    }
                                    else
                                    {

#line 1078
                                        return 2590616169U;
                                    }

#line 1078
                                }
                                else
                                {
                                    if(w_2 < 247U)
                                    {

#line 1082
                                        return 2590349929U;
                                    }
                                    else
                                    {

#line 1084
                                        return 1499830629U;
                                    }

#line 1084
                                }

#line 1084
                            }

#line 1084
                        }
                        else
                        {


                            if(w_2 < 252U)
                            {

#line 1090
                                if(w_2 < 250U)
                                {

#line 1091
                                    if(w_2 < 249U)
                                    {

#line 1092
                                        return 1431656789U;
                                    }
                                    else
                                    {

#line 1094
                                        return 1431590229U;
                                    }

#line 1094
                                }
                                else
                                {
                                    if(w_2 < 251U)
                                    {

#line 1098
                                        return 2590612073U;
                                    }
                                    else
                                    {

#line 1100
                                        return 1499830885U;
                                    }

#line 1100
                                }

#line 1100
                            }
                            else
                            {

                                if(w_2 < 254U)
                                {

#line 1105
                                    if(w_2 < 253U)
                                    {

#line 1106
                                        return 1431656805U;
                                    }
                                    else
                                    {

#line 1108
                                        return 1427395925U;
                                    }

#line 1108
                                }
                                else
                                {
                                    if(w_2 < 255U)
                                    {

#line 1112
                                        return 2573572713U;
                                    }
                                    else
                                    {

#line 1114
                                        return 1431656805U;
                                    }

#line 1114
                                }

#line 1114
                            }

#line 1114
                        }

#line 1114
                    }

#line 1114
                }

#line 1114
            }

#line 1114
        }

#line 1114
    }
    else
    {

#line 1124
        if(w_2 < 384U)
        {

#line 1125
            if(w_2 < 320U)
            {

#line 1126
                if(w_2 < 288U)
                {

#line 1127
                    if(w_2 < 272U)
                    {

#line 1128
                        if(w_2 < 264U)
                        {

#line 1129
                            if(w_2 < 260U)
                            {

#line 1130
                                if(w_2 < 258U)
                                {

#line 1131
                                    if(w_2 < 257U)
                                    {

#line 1132
                                        return 1700091217U;
                                    }
                                    else
                                    {

#line 1134
                                        return 2778093909U;
                                    }

#line 1134
                                }
                                else
                                {
                                    if(w_2 < 259U)
                                    {

#line 1138
                                        return 2795133285U;
                                    }
                                    else
                                    {

#line 1140
                                        return 1363481616U;
                                    }

#line 1140
                                }

#line 1140
                            }
                            else
                            {

                                if(w_2 < 262U)
                                {

#line 1145
                                    if(w_2 < 261U)
                                    {

#line 1146
                                        return 1363481680U;
                                    }
                                    else
                                    {

#line 1148
                                        return 2795137445U;
                                    }

#line 1148
                                }
                                else
                                {
                                    if(w_2 < 263U)
                                    {

#line 1152
                                        return 2795137381U;
                                    }
                                    else
                                    {

#line 1154
                                        return 1704352101U;
                                    }

#line 1154
                                }

#line 1154
                            }

#line 1154
                        }
                        else
                        {


                            if(w_2 < 268U)
                            {

#line 1160
                                if(w_2 < 266U)
                                {

#line 1161
                                    if(w_2 < 265U)
                                    {

#line 1162
                                        return 1704352101U;
                                    }
                                    else
                                    {

#line 1164
                                        return 1431655765U;
                                    }

#line 1164
                                }
                                else
                                {
                                    if(w_2 < 267U)
                                    {

#line 1168
                                        return 2795137446U;
                                    }
                                    else
                                    {

#line 1170
                                        return 1704352101U;
                                    }

#line 1170
                                }

#line 1170
                            }
                            else
                            {

                                if(w_2 < 270U)
                                {

#line 1175
                                    if(w_2 < 269U)
                                    {

#line 1176
                                        return 1363481941U;
                                    }
                                    else
                                    {

#line 1178
                                        return 1704352101U;
                                    }

#line 1178
                                }
                                else
                                {
                                    if(w_2 < 271U)
                                    {

#line 1182
                                        return 1363481685U;
                                    }
                                    else
                                    {

#line 1184
                                        return 1431656805U;
                                    }

#line 1184
                                }

#line 1184
                            }

#line 1184
                        }

#line 1184
                    }
                    else
                    {



                        if(w_2 < 280U)
                        {

#line 1191
                            if(w_2 < 276U)
                            {

#line 1192
                                if(w_2 < 274U)
                                {

#line 1193
                                    if(w_2 < 273U)
                                    {

#line 1194
                                        return 1431659861U;
                                    }
                                    else
                                    {

#line 1196
                                        return 2522441305U;
                                    }

#line 1196
                                }
                                else
                                {
                                    if(w_2 < 275U)
                                    {

#line 1200
                                        return 2523506265U;
                                    }
                                    else
                                    {

#line 1202
                                        return 1158960404U;
                                    }

#line 1202
                                }

#line 1202
                            }
                            else
                            {

                                if(w_2 < 278U)
                                {

#line 1207
                                    if(w_2 < 277U)
                                    {

#line 1208
                                        return 1159025941U;
                                    }
                                    else
                                    {

#line 1210
                                        return 1159025941U;
                                    }

#line 1210
                                }
                                else
                                {
                                    if(w_2 < 279U)
                                    {

#line 1214
                                        return 1159025941U;
                                    }
                                    else
                                    {

#line 1216
                                        return 1159025941U;
                                    }

#line 1216
                                }

#line 1216
                            }

#line 1216
                        }
                        else
                        {


                            if(w_2 < 284U)
                            {

#line 1222
                                if(w_2 < 282U)
                                {

#line 1223
                                    if(w_2 < 281U)
                                    {

#line 1224
                                        return 1158960405U;
                                    }
                                    else
                                    {

#line 1226
                                        return 2523506281U;
                                    }

#line 1226
                                }
                                else
                                {
                                    if(w_2 < 283U)
                                    {

#line 1230
                                        return 1448699481U;
                                    }
                                    else
                                    {

#line 1232
                                        return 1163224409U;
                                    }

#line 1232
                                }

#line 1232
                            }
                            else
                            {

                                if(w_2 < 286U)
                                {

#line 1237
                                    if(w_2 < 285U)
                                    {

#line 1238
                                        return 1091851541U;
                                    }
                                    else
                                    {

#line 1240
                                        return 1448715881U;
                                    }

#line 1240
                                }
                                else
                                {
                                    if(w_2 < 287U)
                                    {

#line 1244
                                        return 1159026009U;
                                    }
                                    else
                                    {

#line 1246
                                        return 1448715881U;
                                    }

#line 1246
                                }

#line 1246
                            }

#line 1246
                        }

#line 1246
                    }

#line 1246
                }
                else
                {

#line 1253
                    if(w_2 < 304U)
                    {

#line 1254
                        if(w_2 < 296U)
                        {

#line 1255
                            if(w_2 < 292U)
                            {

#line 1256
                                if(w_2 < 290U)
                                {

#line 1257
                                    if(w_2 < 289U)
                                    {

#line 1258
                                        return 2572576089U;
                                    }
                                    else
                                    {

#line 1260
                                        return 1413828884U;
                                    }

#line 1260
                                }
                                else
                                {
                                    if(w_2 < 291U)
                                    {

#line 1264
                                        return 2505463125U;
                                    }
                                    else
                                    {

#line 1266
                                        return 2576770649U;
                                    }

#line 1266
                                }

#line 1266
                            }
                            else
                            {

                                if(w_2 < 294U)
                                {

#line 1271
                                    if(w_2 < 293U)
                                    {

#line 1272
                                        return 1413828884U;
                                    }
                                    else
                                    {

#line 1274
                                        return 1431655701U;
                                    }

#line 1274
                                }
                                else
                                {
                                    if(w_2 < 295U)
                                    {

#line 1278
                                        return 2505463129U;
                                    }
                                    else
                                    {

#line 1280
                                        return 2572576089U;
                                    }

#line 1280
                                }

#line 1280
                            }

#line 1280
                        }
                        else
                        {


                            if(w_2 < 300U)
                            {

#line 1286
                                if(w_2 < 298U)
                                {

#line 1287
                                    if(w_2 < 297U)
                                    {

#line 1288
                                        return 2576770649U;
                                    }
                                    else
                                    {

#line 1290
                                        return 2576770649U;
                                    }

#line 1290
                                }
                                else
                                {
                                    if(w_2 < 299U)
                                    {

#line 1294
                                        return 2576770649U;
                                    }
                                    else
                                    {

#line 1296
                                        return 2572576345U;
                                    }

#line 1296
                                }

#line 1296
                            }
                            else
                            {

                                if(w_2 < 302U)
                                {

#line 1301
                                    if(w_2 < 301U)
                                    {

#line 1302
                                        return 2572576345U;
                                    }
                                    else
                                    {

#line 1304
                                        return 1431725401U;
                                    }

#line 1304
                                }
                                else
                                {
                                    if(w_2 < 303U)
                                    {

#line 1308
                                        return 1431655769U;
                                    }
                                    else
                                    {

#line 1310
                                        return 1413829909U;
                                    }

#line 1310
                                }

#line 1310
                            }

#line 1310
                        }

#line 1310
                    }
                    else
                    {



                        if(w_2 < 312U)
                        {

#line 1317
                            if(w_2 < 308U)
                            {

#line 1318
                                if(w_2 < 306U)
                                {

#line 1319
                                    if(w_2 < 305U)
                                    {

#line 1320
                                        return 1431638293U;
                                    }
                                    else
                                    {

#line 1322
                                        return 1717986649U;
                                    }

#line 1322
                                }
                                else
                                {
                                    if(w_2 < 307U)
                                    {

#line 1326
                                        return 1700091225U;
                                    }
                                    else
                                    {

#line 1328
                                        return 1431376149U;
                                    }

#line 1328
                                }

#line 1328
                            }
                            else
                            {

                                if(w_2 < 310U)
                                {

#line 1333
                                    if(w_2 < 309U)
                                    {

#line 1334
                                        return 1717982553U;
                                    }
                                    else
                                    {

#line 1336
                                        return 1431654677U;
                                    }

#line 1336
                                }
                                else
                                {
                                    if(w_2 < 311U)
                                    {

#line 1340
                                        return 1717982553U;
                                    }
                                    else
                                    {

#line 1342
                                        return 1431638293U;
                                    }

#line 1342
                                }

#line 1342
                            }

#line 1342
                        }
                        else
                        {


                            if(w_2 < 316U)
                            {

#line 1348
                                if(w_2 < 314U)
                                {

#line 1349
                                    if(w_2 < 313U)
                                    {

#line 1350
                                        return 1700091225U;
                                    }
                                    else
                                    {

#line 1352
                                        return 1717986922U;
                                    }

#line 1352
                                }
                                else
                                {
                                    if(w_2 < 315U)
                                    {

#line 1356
                                        return 1431655701U;
                                    }
                                    else
                                    {

#line 1358
                                        return 1431655769U;
                                    }

#line 1358
                                }

#line 1358
                            }
                            else
                            {

                                if(w_2 < 318U)
                                {

#line 1363
                                    if(w_2 < 317U)
                                    {

#line 1364
                                        return 1717986921U;
                                    }
                                    else
                                    {

#line 1366
                                        return 286331157U;
                                    }

#line 1366
                                }
                                else
                                {
                                    if(w_2 < 319U)
                                    {

#line 1370
                                        return 1431638293U;
                                    }
                                    else
                                    {

#line 1372
                                        return 1431655769U;
                                    }

#line 1372
                                }

#line 1372
                            }

#line 1372
                        }

#line 1372
                    }

#line 1372
                }

#line 1372
            }
            else
            {

#line 1380
                if(w_2 < 352U)
                {

#line 1381
                    if(w_2 < 336U)
                    {

#line 1382
                        if(w_2 < 328U)
                        {

#line 1383
                            if(w_2 < 324U)
                            {

#line 1384
                                if(w_2 < 322U)
                                {

#line 1385
                                    if(w_2 < 321U)
                                    {

#line 1386
                                        return 2576980377U;
                                    }
                                    else
                                    {

#line 1388
                                        return 2576980377U;
                                    }

#line 1388
                                }
                                else
                                {
                                    if(w_2 < 323U)
                                    {

#line 1392
                                        return 2576980377U;
                                    }
                                    else
                                    {

#line 1394
                                        return 2576980377U;
                                    }

#line 1394
                                }

#line 1394
                            }
                            else
                            {

                                if(w_2 < 326U)
                                {

#line 1399
                                    if(w_2 < 325U)
                                    {

#line 1400
                                        return 2576980377U;
                                    }
                                    else
                                    {

#line 1402
                                        return 2576980377U;
                                    }

#line 1402
                                }
                                else
                                {
                                    if(w_2 < 327U)
                                    {

#line 1406
                                        return 1431673241U;
                                    }
                                    else
                                    {

#line 1408
                                        return 1431655765U;
                                    }

#line 1408
                                }

#line 1408
                            }

#line 1408
                        }
                        else
                        {


                            if(w_2 < 332U)
                            {

#line 1414
                                if(w_2 < 330U)
                                {

#line 1415
                                    if(w_2 < 329U)
                                    {

#line 1416
                                        return 1431655765U;
                                    }
                                    else
                                    {

#line 1418
                                        return 1145324613U;
                                    }

#line 1418
                                }
                                else
                                {
                                    if(w_2 < 331U)
                                    {

#line 1422
                                        return 2576980377U;
                                    }
                                    else
                                    {

#line 1424
                                        return 1431655765U;
                                    }

#line 1424
                                }

#line 1424
                            }
                            else
                            {

                                if(w_2 < 334U)
                                {

#line 1429
                                    if(w_2 < 333U)
                                    {

#line 1430
                                        return 1145328981U;
                                    }
                                    else
                                    {

#line 1432
                                        return 1436129689U;
                                    }

#line 1432
                                }
                                else
                                {
                                    if(w_2 < 335U)
                                    {

#line 1436
                                        return 1431655765U;
                                    }
                                    else
                                    {

#line 1438
                                        return 2576980377U;
                                    }

#line 1438
                                }

#line 1438
                            }

#line 1438
                        }

#line 1438
                    }
                    else
                    {



                        if(w_2 < 344U)
                        {

#line 1445
                            if(w_2 < 340U)
                            {

#line 1446
                                if(w_2 < 338U)
                                {

#line 1447
                                    if(w_2 < 337U)
                                    {

#line 1448
                                        return 1700091217U;
                                    }
                                    else
                                    {

#line 1450
                                        return 2845415830U;
                                    }

#line 1450
                                }
                                else
                                {
                                    if(w_2 < 339U)
                                    {

#line 1454
                                        return 1700091217U;
                                    }
                                    else
                                    {

#line 1456
                                        return 1704565141U;
                                    }

#line 1456
                                }

#line 1456
                            }
                            else
                            {

                                if(w_2 < 342U)
                                {

#line 1461
                                    if(w_2 < 341U)
                                    {

#line 1462
                                        return 1413829969U;
                                    }
                                    else
                                    {

#line 1464
                                        return 1700092309U;
                                    }

#line 1464
                                }
                                else
                                {
                                    if(w_2 < 343U)
                                    {

#line 1468
                                        return 1771674006U;
                                    }
                                    else
                                    {

#line 1470
                                        return 1413829969U;
                                    }

#line 1470
                                }

#line 1470
                            }

#line 1470
                        }
                        else
                        {


                            if(w_2 < 348U)
                            {

#line 1476
                                if(w_2 < 346U)
                                {

#line 1477
                                    if(w_2 < 345U)
                                    {

#line 1478
                                        return 1431655829U;
                                    }
                                    else
                                    {

#line 1480
                                        return 1700108693U;
                                    }

#line 1480
                                }
                                else
                                {
                                    if(w_2 < 347U)
                                    {

#line 1484
                                        return 1704565158U;
                                    }
                                    else
                                    {

#line 1486
                                        return 1413760337U;
                                    }

#line 1486
                                }

#line 1486
                            }
                            else
                            {

                                if(w_2 < 350U)
                                {

#line 1491
                                    if(w_2 < 349U)
                                    {

#line 1492
                                        return 1413829969U;
                                    }
                                    else
                                    {

#line 1494
                                        return 1414878549U;
                                    }

#line 1494
                                }
                                else
                                {
                                    if(w_2 < 351U)
                                    {

#line 1498
                                        return 1414878549U;
                                    }
                                    else
                                    {

#line 1500
                                        return 1431655829U;
                                    }

#line 1500
                                }

#line 1500
                            }

#line 1500
                        }

#line 1500
                    }

#line 1500
                }
                else
                {

#line 1507
                    if(w_2 < 368U)
                    {

#line 1508
                        if(w_2 < 360U)
                        {

#line 1509
                            if(w_2 < 356U)
                            {

#line 1510
                                if(w_2 < 354U)
                                {

#line 1511
                                    if(w_2 < 353U)
                                    {

#line 1512
                                        return 2590401945U;
                                    }
                                    else
                                    {

#line 1514
                                        return 2573624729U;
                                    }

#line 1514
                                }
                                else
                                {
                                    if(w_2 < 355U)
                                    {

#line 1518
                                        return 2573624729U;
                                    }
                                    else
                                    {

#line 1520
                                        return 2572576089U;
                                    }

#line 1520
                                }

#line 1520
                            }
                            else
                            {

                                if(w_2 < 358U)
                                {

#line 1525
                                    if(w_2 < 357U)
                                    {

#line 1526
                                        return 2572576089U;
                                    }
                                    else
                                    {

#line 1528
                                        return 1498768725U;
                                    }

#line 1528
                                }
                                else
                                {
                                    if(w_2 < 359U)
                                    {

#line 1532
                                        return 1431655765U;
                                    }
                                    else
                                    {

#line 1534
                                        return 1427199061U;
                                    }

#line 1534
                                }

#line 1534
                            }

#line 1534
                        }
                        else
                        {


                            if(w_2 < 364U)
                            {

#line 1540
                                if(w_2 < 362U)
                                {

#line 1541
                                    if(w_2 < 361U)
                                    {

#line 1542
                                        return 2590402969U;
                                    }
                                    else
                                    {

#line 1544
                                        return 2572576089U;
                                    }

#line 1544
                                }
                                else
                                {
                                    if(w_2 < 363U)
                                    {

#line 1548
                                        return 1431655765U;
                                    }
                                    else
                                    {

#line 1550
                                        return 1427199061U;
                                    }

#line 1550
                                }

#line 1550
                            }
                            else
                            {

                                if(w_2 < 366U)
                                {

#line 1555
                                    if(w_2 < 365U)
                                    {

#line 1556
                                        return 2573624729U;
                                    }
                                    else
                                    {

#line 1558
                                        return 1431659861U;
                                    }

#line 1558
                                }
                                else
                                {
                                    if(w_2 < 367U)
                                    {

#line 1562
                                        return 1158763589U;
                                    }
                                    else
                                    {

#line 1564
                                        return 1498834265U;
                                    }

#line 1564
                                }

#line 1564
                            }

#line 1564
                        }

#line 1564
                    }
                    else
                    {



                        if(w_2 < 376U)
                        {

#line 1571
                            if(w_2 < 372U)
                            {

#line 1572
                                if(w_2 < 370U)
                                {

#line 1573
                                    if(w_2 < 369U)
                                    {

#line 1574
                                        return 1431655509U;
                                    }
                                    else
                                    {

#line 1576
                                        return 1503028633U;
                                    }

#line 1576
                                }
                                else
                                {
                                    if(w_2 < 371U)
                                    {

#line 1580
                                        return 1431655765U;
                                    }
                                    else
                                    {

#line 1582
                                        return 1503028633U;
                                    }

#line 1582
                                }

#line 1582
                            }
                            else
                            {

                                if(w_2 < 374U)
                                {

#line 1587
                                    if(w_2 < 373U)
                                    {

#line 1588
                                        return 1431393365U;
                                    }
                                    else
                                    {

#line 1590
                                        return 1503028633U;
                                    }

#line 1590
                                }
                                else
                                {
                                    if(w_2 < 375U)
                                    {

#line 1594
                                        return 1162957909U;
                                    }
                                    else
                                    {

#line 1596
                                        return 1435854169U;
                                    }

#line 1596
                                }

#line 1596
                            }

#line 1596
                        }
                        else
                        {


                            if(w_2 < 380U)
                            {

#line 1602
                                if(w_2 < 378U)
                                {

#line 1603
                                    if(w_2 < 377U)
                                    {

#line 1604
                                        return 1503028633U;
                                    }
                                    else
                                    {

#line 1606
                                        return 1162957909U;
                                    }

#line 1606
                                }
                                else
                                {
                                    if(w_2 < 379U)
                                    {

#line 1610
                                        return 1431659865U;
                                    }
                                    else
                                    {

#line 1612
                                        return 1503028633U;
                                    }

#line 1612
                                }

#line 1612
                            }
                            else
                            {

                                if(w_2 < 382U)
                                {

#line 1617
                                    if(w_2 < 381U)
                                    {

#line 1618
                                        return 1146164293U;
                                    }
                                    else
                                    {

#line 1620
                                        return 1162958165U;
                                    }

#line 1620
                                }
                                else
                                {
                                    if(w_2 < 383U)
                                    {

#line 1624
                                        return 1431659865U;
                                    }
                                    else
                                    {

#line 1626
                                        return 1435854233U;
                                    }

#line 1626
                                }

#line 1626
                            }

#line 1626
                        }

#line 1626
                    }

#line 1626
                }

#line 1626
            }

#line 1626
        }
        else
        {

#line 1635
            if(w_2 < 448U)
            {

#line 1636
                if(w_2 < 416U)
                {

#line 1637
                    if(w_2 < 400U)
                    {

#line 1638
                        if(w_2 < 392U)
                        {

#line 1639
                            if(w_2 < 388U)
                            {

#line 1640
                                if(w_2 < 386U)
                                {

#line 1641
                                    if(w_2 < 385U)
                                    {

#line 1642
                                        return 1431654741U;
                                    }
                                    else
                                    {

#line 1644
                                        return 1432704341U;
                                    }

#line 1644
                                }
                                else
                                {
                                    if(w_2 < 387U)
                                    {

#line 1648
                                        return 2506446181U;
                                    }
                                    else
                                    {

#line 1650
                                        return 2506462565U;
                                    }

#line 1650
                                }

#line 1650
                            }
                            else
                            {

                                if(w_2 < 390U)
                                {

#line 1655
                                    if(w_2 < 389U)
                                    {

#line 1656
                                        return 2523239781U;
                                    }
                                    else
                                    {

#line 1658
                                        return 2523240037U;
                                    }

#line 1658
                                }
                                else
                                {
                                    if(w_2 < 391U)
                                    {

#line 1662
                                        return 2523240037U;
                                    }
                                    else
                                    {

#line 1664
                                        return 2523240037U;
                                    }

#line 1664
                                }

#line 1664
                            }

#line 1664
                        }
                        else
                        {


                            if(w_2 < 396U)
                            {

#line 1670
                                if(w_2 < 394U)
                                {

#line 1671
                                    if(w_2 < 393U)
                                    {

#line 1672
                                        return 2523240037U;
                                    }
                                    else
                                    {

#line 1674
                                        return 2506462565U;
                                    }

#line 1674
                                }
                                else
                                {
                                    if(w_2 < 395U)
                                    {

#line 1678
                                        return 1432720741U;
                                    }
                                    else
                                    {

#line 1680
                                        return 1432704357U;
                                    }

#line 1680
                                }

#line 1680
                            }
                            else
                            {

                                if(w_2 < 398U)
                                {

#line 1685
                                    if(w_2 < 397U)
                                    {

#line 1686
                                        return 1431655765U;
                                    }
                                    else
                                    {

#line 1688
                                        return 1364545877U;
                                    }

#line 1688
                                }
                                else
                                {
                                    if(w_2 < 399U)
                                    {

#line 1692
                                        return 1360286036U;
                                    }
                                    else
                                    {

#line 1694
                                        return 2523502185U;
                                    }

#line 1694
                                }

#line 1694
                            }

#line 1694
                        }

#line 1694
                    }
                    else
                    {



                        if(w_2 < 408U)
                        {

#line 1701
                            if(w_2 < 404U)
                            {

#line 1702
                                if(w_2 < 402U)
                                {

#line 1703
                                    if(w_2 < 401U)
                                    {

#line 1704
                                        return 2573571669U;
                                    }
                                    else
                                    {

#line 1706
                                        return 1498764629U;
                                    }

#line 1706
                                }
                                else
                                {
                                    if(w_2 < 403U)
                                    {

#line 1710
                                        return 1431589189U;
                                    }
                                    else
                                    {

#line 1712
                                        return 2590615129U;
                                    }

#line 1712
                                }

#line 1712
                            }
                            else
                            {

                                if(w_2 < 406U)
                                {

#line 1717
                                    if(w_2 < 405U)
                                    {

#line 1718
                                        return 1499829845U;
                                    }
                                    else
                                    {

#line 1720
                                        return 1431655749U;
                                    }

#line 1720
                                }
                                else
                                {
                                    if(w_2 < 407U)
                                    {

#line 1724
                                        return 1158959428U;
                                    }
                                    else
                                    {

#line 1726
                                        return 1499829845U;
                                    }

#line 1726
                                }

#line 1726
                            }

#line 1726
                        }
                        else
                        {


                            if(w_2 < 412U)
                            {

#line 1732
                                if(w_2 < 410U)
                                {

#line 1733
                                    if(w_2 < 409U)
                                    {

#line 1734
                                        return 1431654725U;
                                    }
                                    else
                                    {

#line 1736
                                        return 2590348889U;
                                    }

#line 1736
                                }
                                else
                                {
                                    if(w_2 < 411U)
                                    {

#line 1740
                                        return 1431655765U;
                                    }
                                    else
                                    {

#line 1742
                                        return 1158959429U;
                                    }

#line 1742
                                }

#line 1742
                            }
                            else
                            {

                                if(w_2 < 414U)
                                {

#line 1747
                                    if(w_2 < 413U)
                                    {

#line 1748
                                        return 1499813205U;
                                    }
                                    else
                                    {

#line 1750
                                        return 1158959429U;
                                    }

#line 1750
                                }
                                else
                                {
                                    if(w_2 < 415U)
                                    {

#line 1754
                                        return 1499829589U;
                                    }
                                    else
                                    {

#line 1756
                                        return 1158959429U;
                                    }

#line 1756
                                }

#line 1756
                            }

#line 1756
                        }

#line 1756
                    }

#line 1756
                }
                else
                {

#line 1763
                    if(w_2 < 432U)
                    {

#line 1764
                        if(w_2 < 424U)
                        {

#line 1765
                            if(w_2 < 420U)
                            {

#line 1766
                                if(w_2 < 418U)
                                {

#line 1767
                                    if(w_2 < 417U)
                                    {

#line 1768
                                        return 1431655765U;
                                    }
                                    else
                                    {

#line 1770
                                        return 2590615194U;
                                    }

#line 1770
                                }
                                else
                                {
                                    if(w_2 < 419U)
                                    {

#line 1774
                                        return 1431655765U;
                                    }
                                    else
                                    {

#line 1776
                                        return 1499829849U;
                                    }

#line 1776
                                }

#line 1776
                            }
                            else
                            {

                                if(w_2 < 422U)
                                {

#line 1781
                                    if(w_2 < 421U)
                                    {

#line 1782
                                        return 1158959429U;
                                    }
                                    else
                                    {

#line 1784
                                        return 1431655765U;
                                    }

#line 1784
                                }
                                else
                                {
                                    if(w_2 < 423U)
                                    {

#line 1788
                                        return 1499829849U;
                                    }
                                    else
                                    {

#line 1790
                                        return 1158959429U;
                                    }

#line 1790
                                }

#line 1790
                            }

#line 1790
                        }
                        else
                        {


                            if(w_2 < 428U)
                            {

#line 1796
                                if(w_2 < 426U)
                                {

#line 1797
                                    if(w_2 < 425U)
                                    {

#line 1798
                                        return 1431655765U;
                                    }
                                    else
                                    {

#line 1800
                                        return 1499829849U;
                                    }

#line 1800
                                }
                                else
                                {
                                    if(w_2 < 427U)
                                    {

#line 1804
                                        return 1158959429U;
                                    }
                                    else
                                    {

#line 1806
                                        return 1431655765U;
                                    }

#line 1806
                                }

#line 1806
                            }
                            else
                            {

                                if(w_2 < 430U)
                                {

#line 1811
                                    if(w_2 < 429U)
                                    {

#line 1812
                                        return 1431672409U;
                                    }
                                    else
                                    {

#line 1814
                                        return 1499829849U;
                                    }

#line 1814
                                }
                                else
                                {
                                    if(w_2 < 431U)
                                    {

#line 1818
                                        return 2590615194U;
                                    }
                                    else
                                    {

#line 1820
                                        return 1158959429U;
                                    }

#line 1820
                                }

#line 1820
                            }

#line 1820
                        }

#line 1820
                    }
                    else
                    {



                        if(w_2 < 440U)
                        {

#line 1827
                            if(w_2 < 436U)
                            {

#line 1828
                                if(w_2 < 434U)
                                {

#line 1829
                                    if(w_2 < 433U)
                                    {

#line 1830
                                        return 1430590545U;
                                    }
                                    else
                                    {

#line 1832
                                        return 1430590545U;
                                    }

#line 1832
                                }
                                else
                                {
                                    if(w_2 < 435U)
                                    {

#line 1836
                                        return 1430607189U;
                                    }
                                    else
                                    {

#line 1838
                                        return 1498764629U;
                                    }

#line 1838
                                }

#line 1838
                            }
                            else
                            {

                                if(w_2 < 438U)
                                {

#line 1843
                                    if(w_2 < 437U)
                                    {

#line 1844
                                        return 1498764630U;
                                    }
                                    else
                                    {

#line 1846
                                        return 1498764694U;
                                    }

#line 1846
                                }
                                else
                                {
                                    if(w_2 < 439U)
                                    {

#line 1850
                                        return 1498768790U;
                                    }
                                    else
                                    {

#line 1852
                                        return 1499030934U;
                                    }

#line 1852
                                }

#line 1852
                            }

#line 1852
                        }
                        else
                        {


                            if(w_2 < 444U)
                            {

#line 1858
                                if(w_2 < 442U)
                                {

#line 1859
                                    if(w_2 < 441U)
                                    {

#line 1860
                                        return 1499030934U;
                                    }
                                    else
                                    {

#line 1862
                                        return 1499030934U;
                                    }

#line 1862
                                }
                                else
                                {
                                    if(w_2 < 443U)
                                    {

#line 1866
                                        return 1499030934U;
                                    }
                                    else
                                    {

#line 1868
                                        return 1498768790U;
                                    }

#line 1868
                                }

#line 1868
                            }
                            else
                            {

                                if(w_2 < 446U)
                                {

#line 1873
                                    if(w_2 < 445U)
                                    {

#line 1874
                                        return 1498768790U;
                                    }
                                    else
                                    {

#line 1876
                                        return 1498764694U;
                                    }

#line 1876
                                }
                                else
                                {
                                    if(w_2 < 447U)
                                    {

#line 1880
                                        return 1498764630U;
                                    }
                                    else
                                    {

#line 1882
                                        return 1431655765U;
                                    }

#line 1882
                                }

#line 1882
                            }

#line 1882
                        }

#line 1882
                    }

#line 1882
                }

#line 1882
            }
            else
            {

#line 1890
                if(w_2 < 480U)
                {

#line 1891
                    if(w_2 < 464U)
                    {

#line 1892
                        if(w_2 < 456U)
                        {

#line 1893
                            if(w_2 < 452U)
                            {

#line 1894
                                if(w_2 < 450U)
                                {

#line 1895
                                    if(w_2 < 449U)
                                    {

#line 1896
                                        return 1363477780U;
                                    }
                                    else
                                    {

#line 1898
                                        return 2526697833U;
                                    }

#line 1898
                                }
                                else
                                {
                                    if(w_2 < 451U)
                                    {

#line 1902
                                        return 2522503525U;
                                    }
                                    else
                                    {

#line 1904
                                        return 1448695141U;
                                    }

#line 1904
                                }

#line 1904
                            }
                            else
                            {

                                if(w_2 < 454U)
                                {

#line 1909
                                    if(w_2 < 453U)
                                    {

#line 1910
                                        return 1431655765U;
                                    }
                                    else
                                    {

#line 1912
                                        return 1364542741U;
                                    }

#line 1912
                                }
                                else
                                {
                                    if(w_2 < 455U)
                                    {

#line 1916
                                        return 1363477780U;
                                    }
                                    else
                                    {

#line 1918
                                        return 2522503525U;
                                    }

#line 1918
                                }

#line 1918
                            }

#line 1918
                        }
                        else
                        {


                            if(w_2 < 460U)
                            {

#line 1924
                                if(w_2 < 458U)
                                {

#line 1925
                                    if(w_2 < 457U)
                                    {

#line 1926
                                        return 1431917925U;
                                    }
                                    else
                                    {

#line 1928
                                        return 1431651669U;
                                    }

#line 1928
                                }
                                else
                                {
                                    if(w_2 < 459U)
                                    {

#line 1932
                                        return 1363477780U;
                                    }
                                    else
                                    {

#line 1934
                                        return 2522437989U;
                                    }

#line 1934
                                }

#line 1934
                            }
                            else
                            {

                                if(w_2 < 462U)
                                {

#line 1939
                                    if(w_2 < 461U)
                                    {

#line 1940
                                        return 1431655765U;
                                    }
                                    else
                                    {

#line 1942
                                        return 1363494165U;
                                    }

#line 1942
                                }
                                else
                                {
                                    if(w_2 < 463U)
                                    {

#line 1946
                                        return 2522503525U;
                                    }
                                    else
                                    {

#line 1948
                                        return 1431655765U;
                                    }

#line 1948
                                }

#line 1948
                            }

#line 1948
                        }

#line 1948
                    }
                    else
                    {



                        if(w_2 < 472U)
                        {

#line 1955
                            if(w_2 < 468U)
                            {

#line 1956
                                if(w_2 < 466U)
                                {

#line 1957
                                    if(w_2 < 465U)
                                    {

#line 1958
                                        return 1363494164U;
                                    }
                                    else
                                    {

#line 1960
                                        return 1448696165U;
                                    }

#line 1960
                                }
                                else
                                {
                                    if(w_2 < 467U)
                                    {

#line 1964
                                        return 1364542805U;
                                    }
                                    else
                                    {

#line 1966
                                        return 2522503529U;
                                    }

#line 1966
                                }

#line 1966
                            }
                            else
                            {

                                if(w_2 < 470U)
                                {

#line 1971
                                    if(w_2 < 469U)
                                    {

#line 1972
                                        return 1431655765U;
                                    }
                                    else
                                    {

#line 1974
                                        return 1095042324U;
                                    }

#line 1974
                                }
                                else
                                {
                                    if(w_2 < 471U)
                                    {

#line 1978
                                        return 1431917925U;
                                    }
                                    else
                                    {

#line 1980
                                        return 1363494165U;
                                    }

#line 1980
                                }

#line 1980
                            }

#line 1980
                        }
                        else
                        {


                            if(w_2 < 476U)
                            {

#line 1986
                                if(w_2 < 474U)
                                {

#line 1987
                                    if(w_2 < 473U)
                                    {

#line 1988
                                        return 1431918949U;
                                    }
                                    else
                                    {

#line 1990
                                        return 1363494165U;
                                    }

#line 1990
                                }
                                else
                                {
                                    if(w_2 < 475U)
                                    {

#line 1994
                                        return 1431918949U;
                                    }
                                    else
                                    {

#line 1996
                                        return 1363494165U;
                                    }

#line 1996
                                }

#line 1996
                            }
                            else
                            {

                                if(w_2 < 478U)
                                {

#line 2001
                                    if(w_2 < 477U)
                                    {

#line 2002
                                        return 1431917925U;
                                    }
                                    else
                                    {

#line 2004
                                        return 1095042325U;
                                    }

#line 2004
                                }
                                else
                                {
                                    if(w_2 < 479U)
                                    {

#line 2008
                                        return 1431655781U;
                                    }
                                    else
                                    {

#line 2010
                                        return 2522503529U;
                                    }

#line 2010
                                }

#line 2010
                            }

#line 2010
                        }

#line 2010
                    }

#line 2010
                }
                else
                {

#line 2017
                    if(w_2 < 496U)
                    {

#line 2018
                        if(w_2 < 488U)
                        {

#line 2019
                            if(w_2 < 484U)
                            {

#line 2020
                                if(w_2 < 482U)
                                {

#line 2021
                                    if(w_2 < 481U)
                                    {

#line 2022
                                        return 1431651653U;
                                    }
                                    else
                                    {

#line 2024
                                        return 1499027797U;
                                    }

#line 2024
                                }
                                else
                                {
                                    if(w_2 < 483U)
                                    {

#line 2028
                                        return 1162167557U;
                                    }
                                    else
                                    {

#line 2030
                                        return 1498764629U;
                                    }

#line 2030
                                }

#line 2030
                            }
                            else
                            {

                                if(w_2 < 486U)
                                {

#line 2035
                                    if(w_2 < 485U)
                                    {

#line 2036
                                        return 1515805017U;
                                    }
                                    else
                                    {

#line 2038
                                        return 1162167557U;
                                    }

#line 2038
                                }
                                else
                                {
                                    if(w_2 < 487U)
                                    {

#line 2042
                                        return 1498764629U;
                                    }
                                    else
                                    {

#line 2044
                                        return 1515805017U;
                                    }

#line 2044
                                }

#line 2044
                            }

#line 2044
                        }
                        else
                        {


                            if(w_2 < 492U)
                            {

#line 2050
                                if(w_2 < 490U)
                                {

#line 2051
                                    if(w_2 < 489U)
                                    {

#line 2052
                                        return 1162167557U;
                                    }
                                    else
                                    {

#line 2054
                                        return 1431655749U;
                                    }

#line 2054
                                }
                                else
                                {
                                    if(w_2 < 491U)
                                    {

#line 2058
                                        return 1499027797U;
                                    }
                                    else
                                    {

#line 2060
                                        return 1515870809U;
                                    }

#line 2060
                                }

#line 2060
                            }
                            else
                            {

                                if(w_2 < 494U)
                                {

#line 2065
                                    if(w_2 < 493U)
                                    {

#line 2066
                                        return 1162167621U;
                                    }
                                    else
                                    {

#line 2068
                                        return 1431655749U;
                                    }

#line 2068
                                }
                                else
                                {
                                    if(w_2 < 495U)
                                    {

#line 2072
                                        return 1499026773U;
                                    }
                                    else
                                    {

#line 2074
                                        return 1515805017U;
                                    }

#line 2074
                                }

#line 2074
                            }

#line 2074
                        }

#line 2074
                    }
                    else
                    {



                        if(w_2 < 504U)
                        {

#line 2081
                            if(w_2 < 500U)
                            {

#line 2082
                                if(w_2 < 498U)
                                {

#line 2083
                                    if(w_2 < 497U)
                                    {

#line 2084
                                        return 1157956869U;
                                    }
                                    else
                                    {

#line 2086
                                        return 1162167621U;
                                    }

#line 2086
                                }
                                else
                                {
                                    if(w_2 < 499U)
                                    {

#line 2090
                                        return 1431655749U;
                                    }
                                    else
                                    {

#line 2092
                                        return 1431655765U;
                                    }

#line 2092
                                }

#line 2092
                            }
                            else
                            {

                                if(w_2 < 502U)
                                {

#line 2097
                                    if(w_2 < 501U)
                                    {

#line 2098
                                        return 1499027801U;
                                    }
                                    else
                                    {

#line 2100
                                        return 1515805017U;
                                    }

#line 2100
                                }
                                else
                                {
                                    if(w_2 < 503U)
                                    {

#line 2104
                                        return 1515870810U;
                                    }
                                    else
                                    {

#line 2106
                                        return 84215045U;
                                    }

#line 2106
                                }

#line 2106
                            }

#line 2106
                        }
                        else
                        {


                            if(w_2 < 508U)
                            {

#line 2112
                                if(w_2 < 506U)
                                {

#line 2113
                                    if(w_2 < 505U)
                                    {

#line 2114
                                        return 1162167621U;
                                    }
                                    else
                                    {

#line 2116
                                        return 1162167621U;
                                    }

#line 2116
                                }
                                else
                                {
                                    if(w_2 < 507U)
                                    {

#line 2120
                                        return 1162167621U;
                                    }
                                    else
                                    {

#line 2122
                                        return 1431655749U;
                                    }

#line 2122
                                }

#line 2122
                            }
                            else
                            {

                                if(w_2 < 510U)
                                {

#line 2127
                                    if(w_2 < 509U)
                                    {

#line 2128
                                        return 1431655765U;
                                    }
                                    else
                                    {

#line 2130
                                        return 1431655765U;
                                    }

#line 2130
                                }
                                else
                                {
                                    if(w_2 < 511U)
                                    {

#line 2134
                                        return 1431655765U;
                                    }
                                    else
                                    {

#line 2136
                                        return 1431655765U;
                                    }

#line 2136
                                }

#line 2136
                            }

#line 2136
                        }

#line 2136
                    }

#line 2136
                }

#line 2136
            }

#line 2136
        }

#line 2136
    }

#line 2136
}


#line 41
static float sqrt_exact_0(float x_0)
{

#line 41
    bool _S16;
    if(x_0 >= 5.42101086242752217e-20f)
    {

#line 42
        _S16 = x_0 <= 3.4028234663852886e+38f;

#line 42
    }
    else
    {

#line 42
        _S16 = false;

#line 42
    }

#line 42
    if(!_S16)
    {

#line 43
        return (F32_sqrt((x_0)));
    }
    float y0_0 = (F32_rsqrt((x_0)));
    float g0_0 = x_0 * y0_0;
    float h0_0 = 0.5f * y0_0;
    float r0_0 = (F32_fma((- g0_0), (g0_0), (x_0)));
    float g1_0 = (F32_fma((h0_0), (r0_0), (g0_0)));
    float r1_0 = (F32_fma((- g1_0), (g1_0), (x_0)));
    float g2_0 = (F32_fma((h0_0), (r1_0), (g1_0)));
    uint32_t bits_0 = (F32_asuint((g2_0)));
    float u_up_0 = (U32_asfloat(((bits_0 & 2139095040U) - 192937984U)));

#line 53
    float u_dn_0;
    if((bits_0 & 8388607U) == 0U)
    {

#line 54
        u_dn_0 = 0.5f * u_up_0;

#line 54
    }
    else
    {

#line 54
        u_dn_0 = u_up_0;

#line 54
    }
    float e_1 = (F32_fma((- g2_0), (g2_0), (x_0)));
    float up_0 = e_1 - g2_0 * u_up_0;
    float dn_0 = e_1 + g2_0 * u_dn_0;
    float tu_0 = 0.25f * u_up_0 * u_up_0;
    float td_0 = 0.25f * u_dn_0 * u_dn_0;
    if(up_0 >= tu_0)
    {

#line 61
        return (U32_asfloat((bits_0 + 1U)));
    }
    if(dn_0 < td_0)
    {

#line 64
        return (U32_asfloat((bits_0 - 1U)));
    }
    return g2_0;
}

static float div_exact_0(float a_0, float b_0)
{

#line 70
    float aa_0 = (F32_abs((a_0)));
    float bb_0 = (F32_abs((b_0)));
    float qh_0 = a_0 / b_0;
    float aq_0 = (F32_abs((qh_0)));

#line 73
    bool _S17;
    if(aa_0 >= 7.88860905221011805e-31f)
    {

#line 74
        _S17 = aa_0 <= 1.2676506002282294e+30f;

#line 74
    }
    else
    {

#line 74
        _S17 = false;

#line 74
    }

#line 74
    if(_S17)
    {

#line 74
        if(bb_0 >= 7.88860905221011805e-31f)
        {

#line 74
            _S17 = bb_0 <= 1.2676506002282294e+30f;

#line 74
        }
        else
        {

#line 74
            _S17 = false;

#line 74
        }

#line 74
    }
    else
    {

#line 74
        _S17 = false;

#line 74
    }

#line 74
    if(_S17)
    {

#line 74
        if(aq_0 >= 7.88860905221011805e-31f)
        {

#line 74
            _S17 = aq_0 <= 1.2676506002282294e+30f;

#line 74
        }
        else
        {

#line 74
            _S17 = false;

#line 74
        }

#line 74
    }
    else
    {

#line 74
        _S17 = false;

#line 74
    }

#line 74
    if(!_S17)
    {

#line 75
        return qh_0;
    }
    float y0_1 = 1.0f / bb_0;
    float _S18 = - bb_0;

#line 78
    float e_2 = (F32_fma((_S18), (y0_1), (1.0f)));
    float y1_0 = (F32_fma((y0_1), (e_2), (y0_1)));
    float q0_0 = aa_0 * y1_0;
    float r_2 = (F32_fma((_S18), (q0_0), (aa_0)));
    float q1_0 = (F32_fma((r_2), (y1_0), (q0_0)));
    uint32_t bits_1 = (F32_asuint((q1_0)));
    float u_up_1 = (U32_asfloat(((bits_1 & 2139095040U) - 192937984U)));

#line 84
    float u_dn_1;
    if((bits_1 & 8388607U) == 0U)
    {

#line 85
        u_dn_1 = 0.5f * u_up_1;

#line 85
    }
    else
    {

#line 85
        u_dn_1 = u_up_1;

#line 85
    }
    float r2_0 = (F32_fma((- q1_0), (bb_0), (aa_0)));
    float tu_1 = 0.5f * u_up_1 * bb_0;
    float td_1 = 0.5f * u_dn_1 * bb_0;

#line 88
    uint32_t out_0;

    if(r2_0 >= tu_1)
    {

#line 90
        out_0 = bits_1 + 1U;

#line 90
    }
    else
    {
        if(r2_0 < (- td_1))
        {

#line 93
            out_0 = bits_1 - 1U;

#line 93
        }
        else
        {

#line 93
            out_0 = bits_1;

#line 93
        }

#line 90
    }

#line 97
    return (U32_asfloat((out_0 | (((F32_asuint((a_0))) ^ (F32_asuint((b_0)))) & 2147483648U))));
}


#line 2148
static float rsqrt_zen2_0(float x_1)
{

#line 2149
    uint32_t bits_2 = (F32_asuint((x_1)));
    if((F32_isnan((x_1))))
    {

#line 2151
        return x_1;
    }
    uint32_t ex_0 = (bits_2 >> 23U) & 255U;

#line 2153
    uint32_t par_0;
    if((bits_2 >> 31U) != 0U)
    {

#line 2155
        if(ex_0 == 0U)
        {

#line 2155
            par_0 = 4286578688U;

#line 2155
        }
        else
        {

#line 2155
            par_0 = 4290772992U;

#line 2155
        }

#line 2155
        return (U32_asfloat((par_0)));
    }
    if(ex_0 == 0U)
    {

#line 2158
        return (U32_asfloat((2139095040U)));
    }
    if(bits_2 == 2139095040U)
    {

#line 2161
        return 0.0f;
    }
    uint32_t idx_0 = (bits_2 & 8388607U) >> 11U;
    if((ex_0 & 1U) == 1U)
    {

#line 2164
        par_0 = 0U;

#line 2164
    }
    else
    {

#line 2164
        par_0 = 1U;

#line 2164
    }
    uint32_t n_0 = par_0 * 4096U + idx_0;

    int32_t res_0 = int32_t(((zen2_word_0(n_0 >> 4U)) >> (2U * (n_0 & 15U))) & 3U) - int(1);

#line 2167
    float scale_0;
    if(par_0 == 1U)
    {

#line 2168
        scale_0 = 2.0f;

#line 2168
    }
    else
    {

#line 2168
        scale_0 = 1.0f;

#line 2168
    }
    float t_0 = (float(idx_0) * 2.0f + 1.0f) * 0.0001220703125f;
    float xm_0 = (1.0f + t_0) * scale_0;
    float q_0 = div_exact_0(8192.0f, sqrt_exact_0(xm_0));
    float qr_0 = (F32_floor((q_0 + 0.5f)));



    float r0_1 = (U32_asfloat(((uint32_t(int(126) - ((int32_t(ex_0) - int(127)) >> 1U)) << 23U) | (uint32_t(int32_t(qr_0) + res_0 - int(4096)) << 11U))));
    float r2_1 = x_1 * r0_1;
    float hr_0 = r0_1 * -0.5f;
    float h_0 = (F32_fma((r2_1), (r0_1), (-1.0f)));
    return (F32_fma((hr_0), (h_0), (r0_1)));
}


void _main_0(void* _S19, void* entryPointParams_0, void* globalParams_1)
{

#line 2184
    ComputeThreadVaryingInput * _S20 = (slang_bit_cast<ComputeThreadVaryingInput *>(_S19));

#line 2184
    KernelContext_0 kernelContext_3;

#line 2184
    (&kernelContext_3)->globalParams_0 = (slang_bit_cast<GlobalParams_0*>(globalParams_1));

#line 2184
    Vector<uint32_t, 3>  _S21 = _S20->groupID;

#line 2184
    Vector<uint32_t, 3>  _S22 = _S20->groupThreadID;
    uint32_t _S23 = _S20->groupID.y;

#line 2185
    uint32_t _S24 = pw_0(54U, &kernelContext_3);

#line 2185
    uint32_t e_3 = (_S23 * _S24 + _S21.x) * 256U + _S22.x;

#line 2185
    uint32_t _S25 = pw_0(53U, &kernelContext_3);
    if(e_3 >= _S25)
    {

#line 2187
        return;
    }
    uint32_t i0_2;
    uint32_t i1_2;
    uint32_t i2_2;
    uint32_t i3_2;

#line 2192
    unravel4_0(e_3, 1U, &i0_2, &i1_2, &i2_2, &i3_2, &kernelContext_3);

#line 2192
    uint32_t _S26 = off4_0(1U, i0_2, i1_2, i2_2, i3_2, &kernelContext_3);

#line 2192
    uint32_t _S27 = off4_0(10U, i0_2, i1_2, i2_2, i3_2, &kernelContext_3);

#line 2197
    *(&((&kernelContext_3)->globalParams_0->dst_0)[_S26]) = rsqrt_zen2_0(*(&((&kernelContext_3)->globalParams_0->s0_0)[_S27]));
    return;
}

// [numthreads(256, 1, 1)]
SLANG_PRELUDE_EXPORT
void main_0_Thread(ComputeThreadVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    _main_0(varyingInput, entryPointParams, globalParams);
}
// [numthreads(256, 1, 1)]
SLANG_PRELUDE_EXPORT
void main_0_Group(ComputeVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    ComputeThreadVaryingInput threadInput = {};
    threadInput.groupID = varyingInput->startGroupID;
    for (uint32_t x = 0; x < 256; ++x)
    {
        threadInput.groupThreadID.x = x;
        _main_0(&threadInput, entryPointParams, globalParams);
    }
}
// [numthreads(256, 1, 1)]
SLANG_PRELUDE_EXPORT
void main_0(ComputeVaryingInput* varyingInput, void* entryPointParams, void* globalParams)
{
    ComputeVaryingInput vi = *varyingInput;
    ComputeVaryingInput groupVaryingInput = {};
    for (uint32_t z = vi.startGroupID.z; z < vi.endGroupID.z; ++z)
    {
        groupVaryingInput.startGroupID.z = z;
        for (uint32_t y = vi.startGroupID.y; y < vi.endGroupID.y; ++y)
        {
            groupVaryingInput.startGroupID.y = y;
            for (uint32_t x = vi.startGroupID.x; x < vi.endGroupID.x; ++x)
            {
                groupVaryingInput.startGroupID.x = x;
                main_0_Group(&groupVaryingInput, entryPointParams, globalParams);
            }
        }
    }
}
