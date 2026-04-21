#include <stdio.h>
#include <stdlib.h>

#include "basics.h"

#include "platform_helpers.h"
#include "linux_helpers.c"
#include "sift_parsing.c"

#ifndef PROFILER
#define PROFILER 1
#endif

#include "profiler.h"

#define JUMPSUIT_IMPLEMENTATION
#include "../jumpsuit.h"

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
    char *sift_learn_filename = "data/siftsmall/siftsmall_learn.fvecs";
    char *sift_ground_truth_filename = "data/siftsmall/siftsmall_groundtruth.ivecs";
    char *sift_query_filename = "data/siftsmall/siftsmall_query.fvecs";

    // char *sift_base_filename = "data/sift/sift_base.fvecs";
    // char *sift_learn_filename = "data/sift/sift_learn.fvecs";
    // char *sift_ground_truth_filename = "data/sift/sift_groundtruth.ivecs";
    // char *sift_query_filename = "data/sift/sift_query.fvecs";

    beginProfiler;

    float *base_vectors = 0;
    VecsInfo base_vectors_info = load_fvecs(sift_base_filename, &base_vectors);

    float *learn_vectors = 0;
    VecsInfo learn_vectors_info = 
        load_fvecs(sift_learn_filename, &learn_vectors);

    float *query_vectors = 0;
    VecsInfo query_vectors_info = 
        load_fvecs(sift_query_filename, &query_vectors);

    int *ground_truth_vectors = 0;
    VecsInfo ground_truth_vectors_info = 
        load_ivecs(sift_ground_truth_filename, &ground_truth_vectors);

    // NOTE(fede): k* = 256, m = 8 is recommended when d = 128.
    //      n_bits_per_value  = 8 (when k* = 256)
    IndexPQ index = index_pq_init(base_vectors_info.dimension, 8, 8);

    // index_pq_train(&index, base_vectors, base_vectors_info.n); 
    
    printf("Training %d vectors.\n", learn_vectors_info.n);
    index_pq_train(&index, learn_vectors, learn_vectors_info.n); 

    printf("Adding %d vectors.\n", base_vectors_info.n);
    index_pq_add(&index, base_vectors, base_vectors_info.n);

    free(base_vectors);

    // for (int i = 0; i < index.subvector_dimension; i++) {
    //     printf("%f, ", index.codebook[i]);
    // }
    // printf("\n");

    {
        // int n_neighbours = 100;
        int n_neighbours = 1;
        int n_vectors_search = query_vectors_info.n; 

        IndexPQ_SearchResult search_result = index_pq_search(
                &index,
                query_vectors,
                n_vectors_search,
                n_neighbours);

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
