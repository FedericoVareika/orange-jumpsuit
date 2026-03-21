#ifndef JUMPSUIT_H
#define JUMPSUIT_H

#include "basics.h"

// TODO(fede): for using srand and malloc, 
//   these should be changed for custom random and arenas respectively.
#include <stdlib.h>

// NOTE(fede): Should only be used for sqrt.
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif
    
//////////////////////////////////////////////////////////////////////////////
/// FILE READING

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
/// 2D ARRAY ACCESS


//////////////////////////////////////////////////////////////////////////////
/// VECTOR PROCESSING

typedef struct {
    int dimension;              // = d
    int n_subvectors;           // = m
    int n_bits_per_value;       // = log_2(k*)

    int n_iter;

    ////////////////////////////////////////////////////
    // Calculated automatically based on init params:
    
    int centroids_per_page;     // k*
    int subvector_dimension;

    ////////////////////////////////////////////////////

    ////////////////////////////////////////////////////
    // Filled on train and add
    
    int n_vectors;              // = n

                                //   (C_i = [c_1, c_2, ..., c_k*])
    float *codebook;            // = [C_1, C_2, ..., C_m]

                                //   (Idx_i = [idx_1, idx_2, ..., idx_m])
    int *quantized_codes;       // = [Idx_1, idx_2, ..., idx_n]
                                
    ////////////////////////////////////////////////////
} IndexPQ;

IndexPQ index_pq_init(int dimension, int n_subvectors, int n_bits_per_value);

// TODO(fede): this will allocate memory, pass arena to it.
void index_pq_train(
        IndexPQ *index,
        float *vectors,
        int n_vectors,
        bool row_major);

// TODO(fede): this will allocate memory, pass arena to it.
void index_pq_add(
        IndexPQ *index,
        float *vectors,
        int n_vectors,
        bool row_major);

typedef struct {
    int n_vectors;
    int n_neighbours;

    int *indices;
    float *distances;
} IndexPQ_SearchResult;

// TODO(fede): this will allocate memory, pass arena to it.
IndexPQ_SearchResult index_pq_search(
        IndexPQ *index,
        float *vectors,
        int n_vectors,
        int n_neighbours);

// TODO(fede): remove this after dev
#define JUMPSUIT_IMPLEMENTATION
#ifdef JUMPSUIT_IMPLEMENTATION

IndexPQ index_pq_init(int dimension, int n_subvectors, int n_bits_per_value) {
    timeFunction;
    assert(dimension % n_subvectors == 0);

    return (IndexPQ){
        .dimension = dimension,
        .n_subvectors = n_subvectors,
        .n_bits_per_value = n_bits_per_value,
        .n_iter = 25,
        .subvector_dimension = dimension / n_subvectors,
        .centroids_per_page = 1 << n_bits_per_value,
    };
}

#define oj__clear_array(a, n, v)                                               \
    do {                                                                       \
        for (int __i = 0; __i < (n); __i++) {                                  \
            (a)[__i] = (v);                                                    \
        }                                                                      \
    } while (0)

internal float oj__vector_distance2(float *vec1, float *vec2, int dim) {
    float distance2 = 0;
    for (int i = 0; i < dim; i++) {
        float difference = vec1[i] - vec2[i]; 
        distance2 += difference * difference;
    }
    return distance2;
}

internal float oj__randf(float min, float max) {
    float delta = max - min;
    return (((float)rand() / (float)RAND_MAX) * delta + min);
}

internal int oj__randi(int min, int max) {
    int delta = max - min;
    return (int)(((float)rand() / (float)RAND_MAX) * delta + min);
}

internal float *oj__get_vector(float *vecs, int dim, int idx) {
    return &vecs[idx * dim];
}

internal void oj__add_vectors(float *vec1, float *vec2, int dim, float *dest) {
    for (int i = 0; i < dim; i++) {
        dest[i] = vec1[i] + vec2[i];
    }
}

internal void oj__vector_div(float *vec, int dim, float div, float *dest) {
    for (int i = 0; i < dim; i++) {
        dest[i] = vec[i] / div;
    }
}

internal void oj__vector_copy(float *from, float *dest, int dim) {
    for (int i = 0; i < dim; i++) {
        dest[i] = from[i];
    }
}

typedef struct {
    float *vecs; 
    int dim; 
    int n; 
    bool row_major;

    int subvectors_per_vector;
    int subvector_index;
} Subvectors;

internal float *oj__get_subvector_rm(Subvectors subvectors, int index) {
    int adjusted_index = 
        subvectors.subvector_index +
        index * subvectors.subvectors_per_vector;

    int global_float_index = adjusted_index * subvectors.dim;

    return &subvectors.vecs[global_float_index];
}

internal float *oj__get_random_subvector_rm(Subvectors subvectors) {
    return oj__get_subvector_rm(subvectors, oj__randi(0, subvectors.n));
}

// NOTE(fede): This lead to bad results, ended up using kmeans++ instead
internal void oj__get_random_centroids(
        float *centroids,
        int n_centroids,
        Subvectors subvectors,
        int *centroid_init_idxs) {
    for (int i = 0; i < n_centroids; i++) {
        int rand_idx;
        while (true) {
            rand_idx = oj__randi(0, subvectors.n - 1);

            bool is_different = true;
            for (int j = 0; j < i; j++) {
                if (centroid_init_idxs[j] == rand_idx)
                    is_different = false;
            }

            if (is_different) 
                break;
        }

        centroid_init_idxs[i] = rand_idx;

        float *rand_vector = oj__get_subvector_rm(subvectors, rand_idx);
        oj__vector_copy(
                rand_vector,
                &centroids[i * subvectors.dim],
                subvectors.dim);
    }
}

internal int oj__random_from_cumsum(float *cumsum, int n) {
    float x = oj__randf(0, cumsum[n - 1]);
    int i = -1;
    do {
        x -= cumsum[++i];
    } while (x > 0);

    return i;
}

internal int oj__get_subvector_row_cm(Subvectors subvectors) {
    assert(!subvectors.row_major);
    return subvectors.subvector_index * subvectors.dim;
}

/* v[row][col] where row = [1..dim], col = [1..n_vecs] 
 *
 * subvector: 
 *
 * [ v[1][1], v[2][1], ...,   v[index][1]   , ..., v[n_vecs][ 1 ],
 *   ...
 *   ...
 *   ...                    ---------------                     
 *   v[1][a], v[2][a], ..., | v[index][a] | , ..., v[n_vecs][ a ], |
 *   ...                    |    ...      |                        |-> subvector_dim
 *   v[1][b], v[2][b], ..., | v[index][b] | , ..., v[n_vecs][ b ], |
 *   ...                    ---------------
 *   ...                           ^
 *   ...                       subvector
 *   ...
 *   v[1][n], v[2][n], ...,   v[index][n]   , ..., v[n_vecs][dim] ]
 *
 *   a = row_idx = subvectors.subvector_index * subvectors.dim
 *   b = a + (subvectors.dim - 1)
 *   index = i * subvectors.dim
 */
internal void oj__get_subvector_cm(
        Subvectors subvectors,
        int i, float *dest) {
    assert(!subvectors.row_major);

    int row_idx = oj__get_subvector_row_cm(subvectors);
    for (int j = 0; j < subvectors.dim; j++) {
        int value_idx = (row_idx + j) * subvectors.n + i;

        dest[j] = subvectors.vecs[value_idx];
    }
}

internal void oj__get_random_subvector_cm(Subvectors subvectors, float *dest) {
    assert(!subvectors.row_major);
    int i = oj__randi(0, subvectors.n); 
    oj__get_subvector_cm(subvectors, i, dest);
}

// NOTE(fede): 
//  Kmeans++ algorithm based on FAISS implementation:
//  https://github.com/facebookresearch/faiss/blob/main/faiss/impl/ClusteringInitialization.cpp#L197
//
// TODO(fede): this will allocate memory, pass arena to it.
internal void oj__kmeans_plus_plus_init(
        float *centroids,
        int n_centroids,
        Subvectors subvectors,
        float *min_distances2) {
    if (!subvectors.row_major) {
        oj__get_random_subvector_cm(subvectors, &centroids[0]);

        float *cumsum = malloc(sizeof(float) * subvectors.n);
        assert(cumsum);

        oj__clear_array(min_distances2, subvectors.n, 0);

        int row = oj__get_subvector_row_cm(subvectors);

        {
            float *centroid = &centroids[0];

            for (int j = 0; j < subvectors.dim; j++) {
                int row_idx = (row + j) * subvectors.n;
                float *values_row = &subvectors.vecs[row_idx];

                for (int i = 0; i < subvectors.n; i++) { 
                    float delta = centroid[j] - values_row[i];
                    min_distances2[i] += delta * delta;
                }
            }
        }

        for (int c = 1; c < n_centroids; c++) {
            cumsum[0] = min_distances2[0];
            for (int i = 1; i < subvectors.n; i++) {
                cumsum[i] = cumsum[i - 1] + min_distances2[i];
            }

            float *centroid = &centroids[c * subvectors.dim];
            int new_c = oj__random_from_cumsum(cumsum, subvectors.n);

            oj__get_subvector_cm(
                    subvectors,
                    new_c,
                    centroid);

            // STUDY(fede): this is accessing the entire subvector each loop, 
            //      this will probably cause page faults, check.
            for (int i = 0; i < subvectors.n; i++) {
                float centroid_distance2 = 0;
                for (int j = 0; j < subvectors.dim; j++) {
                    int row_idx = (row + j) * subvectors.n;
                    float value = subvectors.vecs[row_idx + i];

                    float delta = centroid[j] - value; 
                    centroid_distance2 += delta * delta;
                }

                min_distances2[i] = min(centroid_distance2, min_distances2[i]);
            }
        }

        free(cumsum);

    } else {
        oj__vector_copy(
                oj__get_random_subvector_rm(subvectors), 
                &centroids[0],
                subvectors.dim);

        float *cumsum = malloc(sizeof(float) * subvectors.n);
        assert(cumsum);

        for (int i = 0; i < subvectors.n; i++) {
            min_distances2[i] = oj__vector_distance2(
                    &centroids[0],
                    oj__get_subvector_rm(subvectors, i),
                    subvectors.dim);
        }


        for (int c = 1; c < n_centroids; c++) {
            cumsum[0] = min_distances2[0];
            for (int i = 1; i < subvectors.n; i++) {
                cumsum[i] = cumsum[i - 1] + min_distances2[i];
            }

            int new_c = oj__random_from_cumsum(cumsum, subvectors.n);

            oj__vector_copy(oj__get_subvector_rm(subvectors, new_c),
                    &centroids[c * subvectors.dim], subvectors.dim);

            for (int i = 0; i < subvectors.n; i++) {
                float centroid_distance2 = 
                    oj__vector_distance2(
                            // &centroids[c], // TODO(fede): check if its ok 
                            &centroids[c * subvectors.dim],
                            oj__get_subvector_rm(subvectors, i),
                            subvectors.dim);

                min_distances2[i] = min(centroid_distance2, min_distances2[i]);
            }
        }

        free(cumsum);
    }
}

internal int oj__get_closest_centroid(
        float *vec,
        int dim, 
        float *centroids,
        int n_centroids) {
    timeFunction;

    float min_d2 = -1;
    int min_d2_centroid_idx = 0;
    int centroid_idx = 0;

    while (centroid_idx < n_centroids) {
        float *centroid = &centroids[centroid_idx * dim];
        float d2 = oj__vector_distance2(vec, centroid, dim);
        assert(d2 >= 0);

        if (min_d2 < 0 || d2 < min_d2) {
            min_d2 = d2;
            min_d2_centroid_idx = centroid_idx;
        }

        centroid_idx++;
    } 

    return min_d2_centroid_idx;
}

internal void oj__get_clusters(
        Subvectors subvectors,
        float *centroids,
        int n_centroids,
        float *cluster_vector_sums,
        int *cluster_amount,
        float *min_distances2) {
    timeFunction;

    float *vec = 0;
    if (!subvectors.row_major) {
        vec = malloc(sizeof(float) * subvectors.dim);
        assert(vec);
    }

    for (int vec_idx = 0; vec_idx < subvectors.n; vec_idx++) {
        if (subvectors.row_major) {
            vec = oj__get_subvector_rm(subvectors, vec_idx);
        } else {
            assert(vec);
            oj__get_subvector_cm(subvectors, vec_idx, vec);
        }

        int closest_centroid_idx = oj__get_closest_centroid(
                vec,
                subvectors.dim,
                centroids,
                n_centroids);

        float *cluster_sum = oj__get_vector(
                cluster_vector_sums,
                subvectors.dim,
                closest_centroid_idx);

        oj__add_vectors(cluster_sum, vec, subvectors.dim, cluster_sum);
        cluster_amount[closest_centroid_idx]++;
    }

    if (!subvectors.row_major) {
        free(vec);
    }
}

internal void oj__update_centroids(
        Subvectors subvectors,
        float *centroids,
        int n_centroids,
        float *cluster_vector_sums,
        int *cluster_amount,
        bool *should_stop) {
    timeFunction;

    // TODO(fede): parametize this
    const float epsilon = 0.00001;
    *should_stop = true;

    for (int i = 0; i < n_centroids; i++) {
        float *centroid = oj__get_vector(centroids, subvectors.dim, i);

        if (cluster_amount[i] > 0) {
            float *new_centroid = oj__get_vector(cluster_vector_sums, subvectors.dim, i);

            oj__vector_div(
                    new_centroid,
                    subvectors.dim,
                    cluster_amount[i],
                    new_centroid);

            if (oj__vector_distance2(
                        centroid,
                        new_centroid,
                        subvectors.dim) > epsilon) {
                *should_stop = false;
            }

            oj__vector_copy(new_centroid, centroid, subvectors.dim);
        } else {
            *should_stop = false;
            float *new_centroid = oj__get_random_subvector_rm(subvectors);
            oj__vector_copy(new_centroid, centroid, subvectors.dim);
        }
    }
}

// TODO(fede): this will allocate memory, pass arena to it.
// NOTE(fede): 
//      cluster vector sums and cluster amount are pre_allocated 
internal void oj__get_kmeans_cluster_centroids(
        Subvectors subvectors,
        float *centroids, 
        int n_centroids,
        int max_iterations,
        float *cluster_vector_sums,
        int *cluster_amount,
        float *subvector_min_distances2) {
    assert(cluster_vector_sums);

    srand(07734); // hello

    {
        timeBlock("Initialize centroids");

        oj__clear_array(cluster_amount, n_centroids, 0);
        int *centroid_init_idxs = cluster_amount;
        oj__kmeans_plus_plus_init(
                centroids,
                n_centroids,
                subvectors,
                subvector_min_distances2);
    }

    bool should_stop = false;

    {
        timeBlock("Kmeans calculate centroids");
        for (int i = 0; !should_stop && i < max_iterations; i++) {
            oj__clear_array(cluster_vector_sums, n_centroids * subvectors.dim, 0);
            oj__clear_array(cluster_amount, n_centroids, 0);

            oj__get_clusters(
                    subvectors,
                    centroids,
                    n_centroids,
                    cluster_vector_sums,
                    cluster_amount,
                    subvector_min_distances2);

            oj__update_centroids(
                    subvectors,
                    centroids,
                    n_centroids,
                    cluster_vector_sums,
                    cluster_amount,
                    &should_stop);
        }
    }
}

// TODO(fede): this will allocate memory, pass arena to it.
void index_pq_train(
        IndexPQ *index,
        float *vectors,
        int n_vectors,
        bool row_major) {
    timeBandwidth(__func__, n_vectors * index->dimension * sizeof(float));

    //            (C_i = [c_1, c_2, ..., c_k*])
    // codebook = [C_1, C_2, ..., C_m]
    
    int codebook_size =
            sizeof(float) *
            index->subvector_dimension *
            index->centroids_per_page * 
            index->n_subvectors;

    // NOTE(fede): Allow user to query length and allocate this themselves
    if (!index->codebook)
        index->codebook = malloc(codebook_size);

    assert(index->codebook);

    Subvectors subvectors = {
        .vecs = vectors, 
        .n = n_vectors, 
        .dim = index->subvector_dimension, 
        .subvector_index = 0, 
        .subvectors_per_vector = index->n_subvectors,
        .row_major = row_major,
    };

    float *prealloced_cluster_vector_sums = 
        malloc(index->centroids_per_page * subvectors.dim * sizeof(float));
    assert(prealloced_cluster_vector_sums);

    int *prealloced_cluster_amount = malloc(
            index->centroids_per_page * sizeof(int));
    assert(prealloced_cluster_amount);

    float *preallocated_subvector_distances2 = malloc(
            subvectors.n * sizeof(float));
    assert(preallocated_subvector_distances2);

    for (int i = 0; i < index->n_subvectors; i++) {
        float *centroids = &index->codebook[
            i * index->centroids_per_page * index->subvector_dimension];

        oj__get_kmeans_cluster_centroids(
                subvectors,
                centroids,
                index->centroids_per_page,
                index->n_iter,
                prealloced_cluster_vector_sums,
                prealloced_cluster_amount,
                preallocated_subvector_distances2);

        subvectors.subvector_index++;
    }

    free(prealloced_cluster_vector_sums);
    free(prealloced_cluster_amount);
}

internal void oj__set_vector_codes(
        Subvectors subvectors,
        float *centroids, 
        int n_centroids,
        int *quantized_codes) {

    int code_pos = subvectors.subvector_index;
    for (int i = 0; i < subvectors.n; i++) {
        float *vec = oj__get_subvector_rm(subvectors, i);

        int closest_centroid_idx = oj__get_closest_centroid(
                vec, subvectors.dim, centroids, n_centroids);

        quantized_codes[code_pos] = closest_centroid_idx;

        code_pos += subvectors.subvectors_per_vector;
    }
}

// TODO(fede): this will allocate memory, pass arena to it.
void index_pq_add(
        IndexPQ *index,
        float *vectors,
        int n_vectors,
        bool row_major) {
    timeFunction;

    // TODO(fede): allow multiple adds
    assert(index->n_vectors == 0);
    index->n_vectors = n_vectors;

    if (!index->quantized_codes)
        index->quantized_codes = 
            malloc(sizeof(int) * index->n_subvectors * n_vectors);

    assert(index->quantized_codes);

    if (!row_major) {
        float *subvector = malloc(sizeof(float) * index->subvector_dimension);

        Subvectors subvectors = {
            .vecs = vectors,
            .dim = index->subvector_dimension,
            .n = n_vectors,
            .subvector_index = 0,
            .subvectors_per_vector = index->n_subvectors,
            .row_major = row_major,
        };

        for (int j = 0; j < index->n_subvectors; j++) {
            float *centroids = &index->codebook[
                j * index->subvector_dimension * index->centroids_per_page];

            for (int i = 0; i < n_vectors; i++) {
                oj__get_subvector_cm(subvectors, i, subvector);

                int code = oj__get_closest_centroid(
                        subvector,
                        subvectors.dim,
                        centroids, 
                        index->centroids_per_page);

                index->quantized_codes[i * index->n_subvectors + j] = code;
            }

            subvectors.subvector_index++;
        }

        free(subvector);
    } else {
        for (int i = 0; i < n_vectors; i++) {
            float *vector = &vectors[i * index->dimension];
            int *vector_codes = &index->quantized_codes[i * index->n_subvectors];

            for (int j = 0; j < index->n_subvectors; j++) {
                float *subvector = &vector[j * index->subvector_dimension];

                float *centroids = &index->codebook[
                    j * index->subvector_dimension * index->centroids_per_page];
        
                int code = oj__get_closest_centroid(
                        subvector,
                        index->subvector_dimension,
                        centroids,
                        index->centroids_per_page);

                vector_codes[j] = code;
            }
        }
    }

    /*
    for (int i = 0; i < index->n_subvectors; i++) {
        float *centroids = &index->codebook[
            i * index->centroids_per_page * index->subvector_dimension];

        VectorsInput subvectors = {
            .vecs = vectors,
            .n = n_vectors,
            .dim = index->subvector_dimension, 
            .vector_offset = i,
            .vector_pitch = index->dimension / index->subvector_dimension,
        };

        oj__set_vector_codes(subvectors, 
                         centroids,
                         index->centroids_per_page,
                         index->quantized_codes);
    }
    */
}

void oj__insert_ordered(
        IndexPQ_SearchResult *search_result,
        int new_index,
        float new_d2) {
    int insert_at = -1;
    for (int i = 0; i < search_result->n_neighbours; i++) {
        if (search_result->distances[i] == -1 || 
                new_d2 < search_result->distances[i]) {
            insert_at = i;
            break;
        }
    }

    if (insert_at == -1) 
        return;

    for (int i = search_result->n_neighbours - 1;
            i > insert_at;
            i--) {
        search_result->distances[i] = search_result->distances[i - 1];
        search_result->indices[i] = search_result->indices[i - 1];
    }

    search_result->distances[insert_at] = new_d2;
    search_result->indices[insert_at] = new_index;
}

// TODO(fede): this will allocate memory, pass arena to it.
IndexPQ_SearchResult index_pq_search(
        IndexPQ *index,
        float *vectors,
        int n_vectors,
        int n_neighbours) {
    IndexPQ_SearchResult result = {
        .n_vectors = n_vectors,
        .n_neighbours = n_neighbours, 
    };

    result.distances = malloc(sizeof(float) * n_vectors * n_neighbours);
    assert(result.distances);
    oj__clear_array(result.distances, n_vectors * n_neighbours, -1);

    result.indices = malloc(sizeof(int) * n_vectors * n_neighbours);
    assert(result.indices);
    oj__clear_array(result.indices, n_vectors * n_neighbours, -1);

    float *distance_lookup = malloc(
            sizeof(float) * index->n_subvectors * index->centroids_per_page);
    assert(result.indices);
    oj__clear_array(
            result.indices,
            index->n_subvectors * index->centroids_per_page,
            0);

    for (int i = 0; i < n_vectors; i++) {
        float *vector = &vectors[i * index->dimension];

        IndexPQ_SearchResult vector_result = result;
        vector_result.distances = &result.distances[i * n_neighbours];
        vector_result.indices = &result.indices[i * n_neighbours];

        {   // Fill up the distance lookup
            float *subvector = vector;
            for (int i = 0; i < index->n_subvectors; i++) {
                for (int j = 0; j < index->centroids_per_page; j++) {
                    int codebook_idx = i * index->centroids_per_page + j;

                    float *centroid = &index->codebook[
                        codebook_idx * index->subvector_dimension];

                    distance_lookup[codebook_idx] = oj__vector_distance2(
                            subvector,
                            centroid,
                            index->subvector_dimension);
                }
                subvector = &subvector[index->subvector_dimension];
            }
        }

        for (int comparing_vector_idx = 0;
                comparing_vector_idx < index->n_vectors;
                comparing_vector_idx++) {

            // qc[i * m] -> codes of vector i
            int *comparing_vector_qc = &index->quantized_codes[
                comparing_vector_idx * index->n_subvectors];

            float *subvector = vector;
            float distance2 = 0; 
                
            // calculate distance^2 between vector and comparing_vector
            for (int comparing_vector_subvector_idx = 0;
                    comparing_vector_subvector_idx < index->n_subvectors; 
                    comparing_vector_subvector_idx++) {
                
                int cmp_subvector_code = 
                    comparing_vector_qc[comparing_vector_subvector_idx];

                int codebook_idx = cmp_subvector_code +
                    comparing_vector_subvector_idx * index->centroids_per_page; 

                distance2 += distance_lookup[codebook_idx];

                subvector = &subvector[index->subvector_dimension];
            }

            oj__insert_ordered(
                    &vector_result, 
                    comparing_vector_idx,
                    distance2);
        }
    }

    free(distance_lookup);

    // for (int i = 0; i < result.n_vectors * result.n_neighbours; i++) {
    //     if (result.distances[i] < 0) break;
    //     result.distances[i] = sqrt(result.distances[i]);
    // }

    return result;
}

#endif // JUMPSUIT_IMPLEMENTATION

#ifdef __cplusplus
}
#endif

#endif // JUMPSUIT_H
