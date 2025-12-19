#pragma once

#include <cassert>

// define an ASSUME(...) function-style macro so we only need to detect compilers
// in one place

// Comment this out if you don't want assumptions to possibly evaluate.
// This may happen for implementations based on unreachable() functions.
#define DANGEROUS_BEHAVIOR_ASSUMPTIONS_ALLOWED_TO_EVALUATE 1

// preferred option: C++ standard attribute
// #ifdef __has_cpp_attribute
//   #if __has_cpp_attribute(assume) >= 202207L
//     #define DENOX_ASSUME(...) [[assume(__VA_ARGS__)]]
//   #endif
// #endif


// first fallback: compiler intrinsics/attributes for assumptions
#ifndef DENOX_ASSUME
  #if defined(__clang__)
    #define DENOX_ASSUME(...) do { __builtin_assume(__VA_ARGS__); } while(0)
  #elif defined(_MSC_VER)
    #define DENOX_ASSUME(...) do { __assume(__VA_ARGS__); } while(0)
  #elif defined(__GNUC__)
    #if __GNUC__ >= 13
      #define DENOX_ASSUME(...) __attribute__((__assume__(__VA_ARGS__)))
    #endif
  #endif
#endif
// second fallback: possibly evaluating uses of unreachable()
#if !defined(DENOX_ASSUME) && defined(DANGEROUS_BEHAVIOR_ASSUMPTIONS_ALLOWED_TO_EVALUATE)
  #if defined(__GNUC__)
    #define DENOX_ASSUME(...) do { if (!bool(__VA_ARGS__)) __builtin_unreachable(); } while(0)
  #elif __cpp_lib_unreachable >= 202202L
    #include <utility>
    #define DENOX_ASSUME(...) do { if (!bool(__VA_ARGS__)) ::std::unreachable(); ) while(0)
  #endif
#endif
// last fallback: define macro as doing nothing
#ifndef DENOX_ASSUME
  #define DENOX_ASSUME(...)
#endif

#define DENOX_ASSERT_ASSUME(c) do { assert(c); DENOX_ASSUME(c); } while(0)
