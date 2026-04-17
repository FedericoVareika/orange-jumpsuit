#include <stdio.h>
#include <stdlib.h>

#define JUMPSUIT_IMPLEMENTATION
#include "jumpsuit.h"

internal void
generate_random_clusters(float *vectors, int n_points, int dim, int pitch) {
    for (int i = 0; i < n_points; i++) {
        float *v = (float*)((unsigned char*)vectors + (i * pitch));
        
        // Create two clusters: one around (2,2) and one around (8,8)
        float bias = (i < n_points / 2) ? 2.0f : 8.0f;
        
        for (int d = 0; d < dim; d++) {
            // Random value between bias-1.5 and bias+1.5
            v[d] = bias + (((float)rand() / (float)RAND_MAX) * 3.0f - 1.5f);
        }
    }
}

internal void export_csv(float *vecs, int n_vecs, char *filename) {
    int dim = 2;
    FILE *f = fopen(filename, "w");
    if (f) {
        fprintf(f, "x,y\n"); // Header
        for (int i = 0; i < n_vecs; i++) {
            fprintf(f, "%f,%f\n", vecs[i * dim], vecs[i * dim + 1]);
        }
        fclose(f);
    }
}

int main(void) {
    // unsigned int seed = 69;
    unsigned int seed = 420;
    srand(seed);

    int dim = 2;
    int n_points = 200;
    int k = 3;
    
    int vector_pitch = 1; 
    float *vector_buffer = (float*)malloc(n_points * dim * sizeof(float));
    float *centroids = (float*)malloc(k * dim * sizeof(float));

    generate_random_clusters(vector_buffer, n_points, dim, dim * vector_pitch * sizeof(float));

    VectorsInput vectors = {
        .vecs = vector_buffer,
        .n = n_points,
        .dim = dim,
        .vector_offset = 0,
        .vector_pitch = vector_pitch,
    }; 

    printf("Running K-Means...\n");
    char filename_buffer[100] = {};
    for (int i = 0; i < 10; i++) {
        oj__get_kmeans_cluster_centroids(vectors, centroids, k, i);

        sprintf((char *)filename_buffer, "centroids%d.csv", i);
        export_csv(centroids, k, (char *)filename_buffer);
    }

    export_csv(vector_buffer, n_points, "vectors.csv");

    return 0;
}
