#ifndef BASICS_H
#define BASICS_H

#include <stdint.h>

// TODO(fede): for using srand and malloc, 
//   these should be changed for custom random and arenas respectively.
#include <stdlib.h>

// NOTE(fede): Should only be used for sqrt.
#include <math.h>

#define assert(expression)                                                     \
    if (!(expression)) {                                                       \
        *(int *)0 = 0;                                                         \
    }

#define internal static
#define global static
#define local_persist static

#define kilobytes(value) ((value) * 1024)
#define megabytes(value) (kilobytes(value) * 1024)
#define gigabytes(value) (megabytes(value) * 1024)
#define terabytes(value) (gigabytes(value) * 1024)

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define abs(a) ((a) < 0 ? -(a) : (a))

#define array_count(a) (sizeof((a)) / sizeof((a)[0]))

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32; // NOTE(fede): usually dont use
typedef int64_t i64;

typedef float f32;
typedef double f64;

typedef enum { false, true } bool;

#endif // BASICS_H
