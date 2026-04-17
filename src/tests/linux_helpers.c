#include "../platform_helpers.h"

#include <sys/stat.h>
#include <sys/mman.h>

#include <fcntl.h>
#include <unistd.h>

#include <dlfcn.h>

#include <errno.h>

#include <time.h>

ReadFileResult platform_read_entire_file(char *filename) {
    ReadFileResult result = {};

    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        // handle error
        return (ReadFileResult){};
    }

    struct stat stat_;
    if (stat(filename, &stat_) == -1) {
        // handle error
        close(fd);
        return (ReadFileResult){};
    }

    assert(sizeof(stat_.st_size) == sizeof(u64));
    result.size = stat_.st_size;

    result.memory = mmap(0, result.size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    u64 bytes_read = read(fd, result.memory, result.size);
    if (bytes_read != result.size) {
        // handle error
        close(fd);
        return (ReadFileResult){};
    }

    close(fd);

    return result;
}

void platform_free_file_memory(ReadFileResult file_result) {
    if (file_result.memory) {
        munmap(file_result.memory, file_result.size);
    }
}

bool platform_write_entire_file(char *filename, u64 size, void *memory) {
    /*
     * NOTE(fede): When O_CREAT flag is set, a *mode* flag must be set as well.
     *
     *    In this case:
     *
     *         S_IRWXU -- 00700 user (file owner) has read, write, and
     *                    execute permission
     *
     */

    int fd = open(filename, O_RDWR | O_CREAT, S_IRWXU);
    if (fd == -1) {
        // handle error
        return false;
    }

    u64 bytes_written = write(fd, memory, size);
    if (bytes_written != size) {
        // handle error
        return false;
    }

    close(fd);

    return true;
}

i64 platform_get_performance_counter(void)
{
    i64 ticks = 0;
    struct timespec now;

    clock_gettime(CLOCK_MONOTONIC, &now);
    ticks = now.tv_sec;
    ticks *= 1000000000;
    ticks += now.tv_nsec;

    return ticks;
}

u64 platform_get_performance_frequency(void)
{
    return 1000000000;
}
