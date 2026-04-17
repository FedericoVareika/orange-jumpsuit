#include <stdio.h>
#include <stdlib.h>

#define JUMPSUIT_IMPLEMENTATION
#include "jumpsuit.h"

// Generates 4D vectors clustered in 4 distinct groups to make quantization obvious
internal void
generate_random_clusters_4d(float *vectors, int n_points) {
    for (int i = 0; i < n_points; i++) {
        float *v = &vectors[i * 4];
        
        // Group points into 4 distinct clusters
        int cluster = i % 4;
        float bias1 = (cluster & 1) ? 8.0f : 2.0f; // Bias for dims 0 and 1
        float bias2 = (cluster & 2) ? 8.0f : 2.0f; // Bias for dims 2 and 3
        
        // Subvector 0 (dimensions 0 and 1)
        v[0] = bias1 + (((float)rand() / (float)RAND_MAX) * 3.0f - 1.5f);
        v[1] = bias1 + (((float)rand() / (float)RAND_MAX) * 3.0f - 1.5f);
        
        // Subvector 1 (dimensions 2 and 3)
        v[2] = bias2 + (((float)rand() / (float)RAND_MAX) * 3.0f - 1.5f);
        v[3] = bias2 + (((float)rand() / (float)RAND_MAX) * 3.0f - 1.5f);
    }
}

// Exports all data into a single, analysis-friendly CSV
internal void export_pq_results(float *vecs, int *codes, int n_vecs, int m, char *filename) {
    FILE *f = fopen(filename, "w");
    if (f) {
        // Header aligns perfectly with pandas
        fprintf(f, "x1,y1,x2,y2,code0,code1\n"); 
        for (int i = 0; i < n_vecs; i++) {
            fprintf(f, "%f,%f,%f,%f,%d,%d\n", 
                    vecs[i * 4 + 0], vecs[i * 4 + 1],  // Original dims for subvector 0
                    vecs[i * 4 + 2], vecs[i * 4 + 3],  // Original dims for subvector 1
                    codes[i * m + 0],                  // Quantized code for subvector 0
                    codes[i * m + 1]);                 // Quantized code for subvector 1
        }
        fclose(f);
    }
}

// Exports the codebook centroids to a CSV
internal void export_centroids(IndexPQ *index, char *filename) {
    FILE *f = fopen(filename, "w");
    if (f) {
        // Header aligns with pandas
        fprintf(f, "subvector_idx,centroid_idx,cx,cy\n"); 
        
        for (int m_idx = 0; m_idx < index->n_subvectors; m_idx++) {
            for (int k_idx = 0; k_idx < index->centroids_per_page; k_idx++) {
                
                // Calculate the offset in the flat codebook array
                // codebook layout: [Subvector 0 centroids...] [Subvector 1 centroids...]
                int offset = (m_idx * index->centroids_per_page * index->subvector_dimension) + 
                             (k_idx * index->subvector_dimension);
                
                fprintf(f, "%d,%d,%f,%f\n", 
                        m_idx, 
                        k_idx, 
                        index->codebook[offset + 0], 
                        index->codebook[offset + 1]);
            }
        }
        fclose(f);
    }
}

int main(void) {
    unsigned int seed = 69;
    srand(seed);

    int dim = 4;
    int n_points = 400;
    
    // PQ Configuration
    int m = 2;          // Split 4D vector into two 2D subvectors
    int n_bits = 1;     // 4 centroids per subvector (k* = 2^2 = 4)
    
    // Allocate buffers
    float *vector_buffer = (float*)malloc(n_points * dim * sizeof(float));
    
    generate_random_clusters_4d(vector_buffer, n_points);

    printf("Initializing PQ Index...\n");
    IndexPQ index = index_pq_init(dim, m, n_bits);

    printf("Training PQ Index...\n");
    index_pq_train(&index, vector_buffer, n_points);

    printf("Adding vectors to PQ Index...\n");
    index_pq_add(&index, vector_buffer, n_points);

    printf("Exporting results...\n");
    export_pq_results(vector_buffer, index.quantized_codes, n_points, m, "pq_vectors.csv");

    export_centroids(&index, "pq_centroids.csv");

    int n_vectors_search = 2;
    int n_neighbours = 3;
    IndexPQ_SearchResult search_result = index_pq_search(
            &index, vector_buffer, n_vectors_search, n_neighbours);

    for (int i = 0; i < n_vectors_search; i++) {
        float *vector = &vector_buffer[i * dim];
        int *indices = &search_result.indices[i * n_neighbours];
        float *distances = &search_result.distances[i * n_neighbours];

        printf("For vector (");
        for (int j = 0; j < dim; j++) {
            printf("%.2f", vector[j]);
            if (j < dim - 1)
                printf(", ");
        }
        printf("): \n [ \n");

        for (int j = 0; j < n_neighbours; j++) {
            printf("%d, distance = %.2f\n", indices[j], distances[j]);
        }

        printf("] \n\n");
    }


    return 0;
}

