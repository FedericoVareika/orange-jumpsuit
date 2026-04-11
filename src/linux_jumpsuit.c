#include <stdio.h>
#include <stdlib.h>

#include "basics.h"
#include "linux_helpers.h"

#ifndef PROFILER
#define PROFILER 1
#endif

#include "profiler.h"

#define JUMPSUIT_IMPLEMENTATION
#include "jumpsuit.h"

internal void export_csv(float *vecs, int n_vecs, int dim, char *filename) {
    FILE *f = fopen(filename, "w");
    if (f) {
        for (int i = 0; i < n_vecs; i++) {
            for (int j = 0; j < dim; j++) {
                fprintf(f, "%f", vecs[i * dim + j]);
                if (j < dim - 1)
                    fprintf(f, ",");
                else 
                    fprintf(f, "\n");
            }
        }
        fclose(f);
    }
}

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

    ReadFileResult file = read_entire_file(filename);
    assert(file.memory);
    Buffer buf = {
        .data = file.memory,
        .size = file.size,
    };

    VecsInfo info = get_vecs_info(buf);

    *vectors = malloc(info.n * info.dimension * sizeof(float));
    fill_fvecs(&buf, *vectors, info);

    free_file_memory(file);

    return info;
}

VecsInfo load_ivecs(char *filename, int **vectors) {
    assert(vectors);
    assert(filename);

    ReadFileResult file = read_entire_file(filename);
    assert(file.memory);
    Buffer buf = {
        .data = file.memory,
        .size = file.size,
    };

    VecsInfo info = get_vecs_info(buf);

    *vectors = malloc(info.n * info.dimension * sizeof(float));
    fill_ivecs(&buf, *vectors, info);

    free_file_memory(file);

    return info;
}

float recall_at_r(
        IndexPQ_SearchResult search_result,
        int *ground_truths,
        VecsInfo ground_truths_info,
        int r) {
    float result = 0; 
    for (int i = 0; i < search_result.n_vectors; i++) {
        int ground_truth = ground_truths[
            i * ground_truths_info.dimension];

        int *indices = &search_result.indices[i * search_result.n_neighbours];

        for (int j = 0; j < r; j++) {
            if (indices[j] == ground_truth) {
                result++;
                break;
            }
        }
    }

    return result / (float)search_result.n_vectors;
 }

int main(int argc, char **argv) {
    char *sift_base_filename = "data/siftsmall/siftsmall_base.fvecs";
    char *sift_ground_truth_filename = 
        "data/siftsmall/siftsmall_groundtruth.ivecs";
    char *sift_query_filename = 
        "data/siftsmall/siftsmall_query.fvecs";

    beginProfiler;

    float *base_vectors = 0;
    VecsInfo base_vectors_info = load_fvecs(sift_base_filename, &base_vectors);

    float *query_vectors = 0;
    VecsInfo query_vectors_info = 
        load_fvecs(sift_query_filename, &query_vectors);

    int *ground_truth_vectors = 0;
    VecsInfo ground_truth_vectors_info = 
        load_ivecs(sift_ground_truth_filename, &ground_truth_vectors);

    // NOTE(fede): k* = 256, m = 8 is recommended when d = 128.
    //      n_bits_per_value  = 8 (when k* = 256)
    IndexPQ index = index_pq_init(base_vectors_info.dimension, 8, 8);

    index_pq_train(&index, base_vectors, base_vectors_info.n); 

    index_pq_add(&index, base_vectors, base_vectors_info.n);

    free(base_vectors);

    for (int i = 0; i < index.subvector_dimension; i++) {
        printf("%f, ", index.codebook[i]);
    }
    printf("\n");

    {
        int n_neighbours = query_vectors_info.dimension; 
        int n_vectors_search = query_vectors_info.n; 

        int *search_indices = 
            malloc(sizeof(int) * n_neighbours * n_vectors_search);
        float *search_distances = 
            malloc(sizeof(float) * n_neighbours * n_vectors_search);

        IndexPQ_SearchResult search_result = index_pq_search(
                &index,
                query_vectors,
                n_vectors_search,
                n_neighbours);

        // for (int i = 0; i < 3; i++) {
        //     float *vector = &query_vectors[i * base_vectors_info.dimension];
        //     int *indices = &search_result.indices[i * n_neighbours];
        //     float *distances = &search_result.distances[i * n_neighbours];

        //    printf("For vector (");
        //     for (int j = 0; j < base_vectors_info.dimension; j++) {
        //         printf("%.2f", vector[j]);
        //         if (j != base_vectors_info.dimension)
        //             printf(", ");
        //     }
        //     printf("): \n [ \n");

        //     for (int j = 0; j < 3; j++) {
        //         printf("    %d, distance = %.2f\n", indices[j], distances[j]);
        //     }

        //     printf("] \n\n");

        //     if (i < ground_truth_vectors_info.n) {
        //         printf("Ground truth: [\n");

        //         int *ground_truths = &ground_truth_vectors[
        //             i * ground_truth_vectors_info.dimension];
        //         for (int j = 0; j < 3; j++) {
        //             printf("    %d\n", ground_truths[j]);
        //         }

        //         printf("] \n\n");
        //     } else {
        //         printf("Ground truth is not available for vector n: %d\n", i);
        //     }
        // }
        
        printf("%6s | Result\n", "GT");
        for (int i = 0; i < search_result.n_vectors; i++) {
            int vector_search_result = 
                search_result.indices[i * search_result.n_neighbours];
            int vector_ground_truth =
                ground_truth_vectors[i * ground_truth_vectors_info.dimension];

            printf("%6d | %d\n", vector_ground_truth, vector_search_result);
        }

        printf("\nRecall@1: %.f%% \n\n",
                100 * recall_at_r(
                    search_result,
                    ground_truth_vectors,
                    ground_truth_vectors_info, 1));
    }


    end_and_print_profiler();

    return 0;
}

ProfilerEndOfCompilationUnit;
