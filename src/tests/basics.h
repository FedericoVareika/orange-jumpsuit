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


//////////////////////////////////////////////////////////////////////////////
/// BUFFER READING

typedef struct {
    u8 *data;
    int cursor; 
    int size;
} Buffer;

global u8 buffer_get(Buffer *buf) {
    if (buf->cursor >= buf->size) 
        return 0;
    return buf->data[buf->cursor++];
}

global void buffer_skip(Buffer *buf, int n) {
    assert(buf->cursor + n <= buf->size);
    buf->cursor += n;
}

global f32 buffer_get_f32(Buffer *buf) {
    assert(buf->cursor % sizeof(f32) == 0);
    f32 result = *(f32 *)(&buf->data[buf->cursor]);
    buffer_skip(buf, sizeof(f32));
    return result;
}

global int buffer_get_int(Buffer *buf) {
    assert(buf->cursor % sizeof(int) == 0);
    int result = *(int *)(&buf->data[buf->cursor]);
    buffer_skip(buf, sizeof(int));
    return result;
}

//////////////////////////////////////////////////////////////////////////////

#endif // BASICS_H
