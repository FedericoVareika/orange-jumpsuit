#include <stdlib.h>
#include "platform_helpers.h"

typedef struct {
    int n, dimension;
} VecsInfo;

VecsInfo get_vecs_info(Buffer buffer) {
    // NOTE(fede): Assume that this format will give the same dimension for
    //              every vector, either way it would be a faulty state.  
    VecsInfo result;
    result.dimension = buffer_get_int(&buffer);
    int sizeof_row = sizeof(int) + result.dimension * sizeof(float);
    assert(buffer.size % sizeof_row == 0);
    result.n = buffer.size / sizeof_row;

    return result;
}

void fill_fvecs(
        Buffer *buffer,
        float *vectors,
        VecsInfo info) {

    int global_vector_idx = 0;
    for (int i = 0; i < info.n; i++) {
        int current_dimension = buffer_get_int(buffer);
        assert(current_dimension == info.dimension);

        for (int j = 0; j < info.dimension; j++) {
            vectors[global_vector_idx++] = buffer_get_f32(buffer);
        }
    }

    assert(global_vector_idx == info.n * info.dimension);
}

void fill_ivecs(
        Buffer *buffer,
        int *vectors,
        VecsInfo info) {

    int global_vector_idx = 0;
    for (int i = 0; i < info.n; i++) {
        int current_dimension = buffer_get_int(buffer);
        assert(current_dimension == info.dimension);

        for (int j = 0; j < info.dimension; j++) {
            vectors[global_vector_idx++] = buffer_get_int(buffer);
        }
    }

    assert(global_vector_idx == info.n * info.dimension);
}

VecsInfo load_fvecs(char *filename, float **vectors) {
    assert(vectors);
    assert(filename);

    ReadFileResult file = platform_read_entire_file(filename);
    assert(file.memory);
    Buffer buf = {
        .data = file.memory,
        .size = file.size,
    };

    VecsInfo info = get_vecs_info(buf);

    *vectors = malloc(info.n * info.dimension * sizeof(float));
    fill_fvecs(&buf, *vectors, info);

    platform_free_file_memory(file);

    return info;
}

VecsInfo load_ivecs(char *filename, int **vectors) {
    assert(vectors);
    assert(filename);

    ReadFileResult file = platform_read_entire_file(filename);
    assert(file.memory);
    Buffer buf = {
        .data = file.memory,
        .size = file.size,
    };

    VecsInfo info = get_vecs_info(buf);

    *vectors = malloc(info.n * info.dimension * sizeof(float));
    fill_ivecs(&buf, *vectors, info);

    platform_free_file_memory(file);

    return info;
}

