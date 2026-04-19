#ifndef PROFILER
#define PROFILER 0
#endif

#if PROFILER

#include "platform_helpers.h"

typedef struct {
    const char *label;

    u64 inclusive_elapsed_time;
    u64 exclusive_elapsed_time;

    u64 hit_count;
    u64 processed_byte_count;
} ProfileAnchor;

typedef struct {
    ProfileAnchor anchors[4096];

    u64 start_time;
    u64 end_time;
} Profiler;

static Profiler global_profiler;
static u32 global_profiler_parent;

typedef struct {
    const char *label;
    u32 anchor_index;
    u32 parent_index;

    u64 start_tsc;
    u64 old_inclusive_elapsed_time;
} ProfileBlock;

void profile_block_destructor(ProfileBlock *profile_block_ptr) {
    u32 anchor_index = profile_block_ptr->anchor_index;
    u32 parent_index = profile_block_ptr->parent_index;

    global_profiler_parent = parent_index;

    u64 block_start_tsc = profile_block_ptr->start_tsc;
    u64 old_inclusive_elapsed_time =
        profile_block_ptr->old_inclusive_elapsed_time;

    u64 elapsed_time = platform_get_performance_counter() - block_start_tsc;

    ProfileAnchor *anchor = &global_profiler.anchors[anchor_index];
    ProfileAnchor *parent = &global_profiler.anchors[parent_index];

    parent->exclusive_elapsed_time -= elapsed_time;
    anchor->exclusive_elapsed_time += elapsed_time;
    anchor->inclusive_elapsed_time = old_inclusive_elapsed_time + elapsed_time;

    anchor->label = profile_block_ptr->label;
    anchor->hit_count++;
}

ProfileBlock construct_block(const char *anchor_label, u32 anchor_index,
                             u64 byte_count) {
    ProfileAnchor *anchor = &global_profiler.anchors[anchor_index];
    ProfileBlock profile_block = {
        anchor_label,
        anchor_index,
        global_profiler_parent,
        platform_get_performance_counter(),
        anchor->inclusive_elapsed_time,
    };
    anchor->processed_byte_count += byte_count;
    global_profiler_parent = anchor_index;
    return profile_block;
}

#define timeBandwidth(anchor_label, byte_count)                                \
    ProfileBlock __attribute__((unused)) __attribute__((                       \
        __cleanup__(profile_block_destructor))) profile_block##__LINE__ =      \
        construct_block(anchor_label, __COUNTER__ + 1, byte_count)

#define timeBlock(anchor_label) timeBandwidth(anchor_label, 0)

#define timeFunction timeBlock(__func__)

#define beginProfiler                                                          \
    { global_profiler.start_time = platform_get_performance_counter(); }

#define endProfiler                                                            \
    { global_profiler.end_time = platform_get_performance_counter(); }

void print_elapsed_time(u64 total_clocks, u64 timer_freq,
                        ProfileAnchor *anchor) {
    f64 percentage_exclusive =
        (f64)(anchor->exclusive_elapsed_time * 100) / (f64)total_clocks;
    // printf("  %s[%llu]: %llu (%.2f%%", anchor->label, anchor->hit_count,
    printf("  %s[%lu]: %lu (%.2f%%", anchor->label, anchor->hit_count,
           anchor->exclusive_elapsed_time, percentage_exclusive);

    if (anchor->inclusive_elapsed_time != anchor->exclusive_elapsed_time) {
        f64 percentage_inclusive =
            (f64)(anchor->inclusive_elapsed_time * 100) / (f64)total_clocks;
        printf(", %.2f%% w/children", percentage_inclusive);
    }

    if (anchor->processed_byte_count) {
        f64 megabyte = 1024.0f * 1024.0f;

        f64 seconds = (f64)anchor->inclusive_elapsed_time / (f64)timer_freq;
        f64 bytes_per_second = (f64)anchor->processed_byte_count / seconds;
        f64 megabytes = (f64)anchor->processed_byte_count / (f64)megabyte;
        f64 megabytes_per_second = bytes_per_second / megabyte;

        // printf("  %.3fmb at %.2fgb/s", megabytes, gigabytes_per_second);
        printf("  %.3fmb at %.2fmb/s", megabytes, megabytes_per_second);
    }

    printf(")\n");
}

#define end_and_print_profiler()                                               \
    {                                                                          \
        endProfiler;                                                           \
        u64 cpu_freq = platform_get_performance_frequency();                   \
        u64 total_clocks =                                                     \
            global_profiler.end_time - global_profiler.start_time;             \
        f64 total_time = (f64)total_clocks / (f64)cpu_freq;                    \
        printf("Total time: %f (Cpu freq: %lu)\n", total_time, cpu_freq);     \
                                                                               \
        for (int i = 0; i < __COUNTER__; i++) {                                \
            ProfileAnchor anchor = global_profiler.anchors[i + 1];             \
            print_elapsed_time(total_clocks, cpu_freq, &anchor);               \
        }                                                                      \
    }

#define ArrayCount(Array) (sizeof(Array) / sizeof((Array)[0]))
#define ProfilerEndOfCompilationUnit                                           \
    _Static_assert(                                                             \
        __COUNTER__ < ArrayCount(global_profiler.anchors),                     \
        "Number of profile points exceeds size of profiler::Anchors array")

#else

#define timeBandwidth(...)
#define timeBlock(...)
#define timeFunction
#define beginProfiler
#define endProfiler
#define end_and_print_profiler(...)
#define ProfilerEndOfCompilationUnit

#endif

