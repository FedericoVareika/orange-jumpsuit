#ifndef PLATFORM_HELPERS_H
#define PLATFORM_HELPERS_H

// #include "basics.h"

typedef struct {
    u64 size;
    void *memory;
} ReadFileResult;

ReadFileResult platform_read_entire_file(char *filename);
void platform_free_file_memory(ReadFileResult file_result);
bool platform_write_entire_file(char *filename, u64 size, void *memory);

i64 platform_get_performance_counter(void);
u64 platform_get_performance_frequency(void);

#endif // PLATFORM_HELPERS_H
