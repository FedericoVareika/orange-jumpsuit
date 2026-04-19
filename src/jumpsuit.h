#ifndef JUMPSUIT_H
#define JUMPSUIT_H

/////////////////////////////////////////////////////////////////////
// 
// Function specifiers in case library is build/used as a shared library
//
// Copied from: 
//  https://github.com/raysan5/raylib/blob/master/src/raylib.h
//

#if defined(_WIN32)
    #if defined(__TINYC__)
        #define __declspec(x) __attribute__((x))
    #endif
    #if defined(OJ_EXPORT)
        #define OJ_API __declspec(dllexport)     // Building the library as a Win32 shared library (.dll)
    #else
        #define OJ_API __declspec(dllimport)     // Using the library as a Win32 shared library (.dll)
    #endif
#else
    #if defined(BUILD_LIBTYPE_SHARED)
        #define OJ_API __attribute__((visibility("default"))) // Building as a Unix shared library (.so/.dylib)
    #endif
#endif

#ifndef OJ_API
    #define OJ_API  // Functions defined as 'extern' by default (implicit specifiers)
#endif

/////////////////////////////////////////////////////////////////////
// Cross-Platform alligned alloc

#if defined(_WIN32) || defined(_WIN64)
    #include <malloc.h>
    #define oj__aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
    #define oj__aligned_free(ptr) _aligned_free(ptr)
#else
    #define oj__aligned_alloc(alignment, size) aligned_alloc(alignment, size)
    #define oj__aligned_free(ptr) free(ptr)
#endif

/////////////////////////////////////////////////////////////////////
// Dependencies

// NOTE(fede): for using srand, aligned_alloc, and malloc, 
#include <stdlib.h>

// NOTE(fede): Should only be used for sqrt.
#include <math.h>

/////////////////////////////////////////////////////////////////////
// Defines

#ifdef __cplusplus
extern "C" {
#endif

#define assert(expression)                                                     \
    if (!(expression)) {                                                       \
        *(int *)0 = 0;                                                         \
    }

#define internal static
#define global static
#define local_persist static

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define abs(a) ((a) < 0 ? -(a) : (a))

#define array_count(a) (sizeof((a)) / sizeof((a)[0]))

#ifndef oj__bool
typedef enum { oj__false, oj__true } oj__bool;
#endif

    
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

typedef struct {
    int n_vectors;
    int n_neighbours;

    int *indices;
    float *distances;
} IndexPQ_SearchResult;

OJ_API IndexPQ index_pq_init(int dimension, int n_subvectors, int n_bits_per_value);
OJ_API void index_pq_train(IndexPQ *index, float *vectors, int n_vectors);
OJ_API void index_pq_add(IndexPQ *index, float *vectors, int n_vectors);
OJ_API IndexPQ_SearchResult index_pq_search(IndexPQ *index, float *vectors, int n_vectors, int n_neighbours);

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

internal void oj__add_vectors(float *vec1, float *vec2, int dim, float *dest) {
    for (int i = 0; i < dim; i++) {
        dest[i] = vec1[i] + vec2[i];
    }
}

internal inline float oj__vector_dotproduct(float *vec1, float *vec2, int dim) {
    float result = 0;
    for (int i = 0; i < dim; i++) {
        result += vec1[i] * vec2[i];
    }
    return result;
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
    int n;
    int dim;
} Vectors;

internal float *oj__get_vector(Vectors vectors, int index) {
    return &vectors.vecs[vectors.dim * index];
}

internal float *oj__get_random_vector(Vectors vectors) {
    return oj__get_vector(vectors, oj__randi(0, vectors.n));
}

internal float oj__calculate_sqr_norm(Vectors vectors, int i) {
    float sqr_norm = 0;
    for (int j = 0; j < vectors.dim; j++) {
        float v = vectors.vecs[i * vectors.dim + j];
        sqr_norm += v * v;
    }

    return sqr_norm;
}  

#include <cblas.h>

// C = alpha*a*b + beta*c
internal void oj__multiply_vectors_blas(
        Vectors a,
        Vectors t_b,
        Vectors c,
        float alpha, 
        float beta) {
    // timeFunction; 

    float *A = a.vecs;
    float *BT = t_b.vecs;
    float *C = c.vecs;

    int M = a.n; 
    int K = a.dim; 
    int N = t_b.n;

    cblas_sgemm(CblasRowMajor, 
                CblasNoTrans,   // A is normal
                CblasTrans,     // B is provided as B^T, so we "Trans" it back
                M, N, K, 
                alpha, 
                A, K,           // lda: leading dimension of A
                BT, K,          // ldb: leading dimension of BT (it's K)
                beta, 
                C, N);          // ldc: leading dimension of C
}

/*
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

        float *rand_vector = oj__get_subvector(subvectors, rand_idx);
        oj__vector_copy(
                rand_vector,
                &centroids[i * subvectors.dim],
                subvectors.dim);
    }
}
*/

internal int oj__random_from_cumsum(float *cumsum, int n) {
    float x = oj__randf(0, cumsum[n - 1]);
    int i = -1;
    do {
        x -= cumsum[++i];
    } while (x > 0);

    return i;
}

// NOTE(fede): 
//  Kmeans++ algorithm based on FAISS implementation:
//  https://github.com/facebookresearch/faiss/blob/main/faiss/impl/ClusteringInitialization.cpp#L197
//
internal void oj__kmeans_plus_plus_init(
        Vectors centroids,
        Vectors subvectors,
        float *min_distances2,
        float *subvector_sqr_norms) {

    oj__vector_copy(
            oj__get_random_vector(subvectors), 
            oj__get_vector(centroids, 0),
            subvectors.dim);

    float *cumsum = malloc(sizeof(float) * subvectors.n);
    assert(cumsum);

    for (int i = 0; i < subvectors.n; i++) {
        min_distances2[i] = oj__vector_distance2(
                oj__get_vector(centroids, 0),
                oj__get_vector(subvectors, i),
                subvectors.dim);
    }

    for (int c = 1; c < centroids.n; c++) {
        cumsum[0] = min_distances2[0];
        for (int i = 1; i < subvectors.n; i++) {
            cumsum[i] = cumsum[i - 1] + min_distances2[i];
        }

        int new_c = oj__random_from_cumsum(cumsum, subvectors.n);

        oj__vector_copy(
                oj__get_vector(subvectors, new_c),
                oj__get_vector(centroids, c),
                subvectors.dim);

        for (int i = 0; i < subvectors.n; i++) {
            float centroid_distance2 = 
                oj__vector_distance2(
                        oj__get_vector(centroids, c),
                        oj__get_vector(subvectors, i),
                        subvectors.dim);

            min_distances2[i] = min(centroid_distance2, min_distances2[i]);
        }
    }

    free(cumsum);
}

internal int oj__get_closest_centroid(Vectors centroids, float *vec) {
    float min_d2 = -1;
    int min_d2_centroid_idx = 0;
    int centroid_idx = 0;

    while (centroid_idx < centroids.n) {
        float d2 = oj__vector_distance2(
                oj__get_vector(centroids, centroid_idx), vec, centroids.dim);
        assert(d2 >= 0);

        if (min_d2 < 0 || d2 < min_d2) {
            min_d2 = d2;
            min_d2_centroid_idx = centroid_idx;
        }

        centroid_idx++;
    } 

    return min_d2_centroid_idx;
}

internal void oj__calculate_sqr_norms(Vectors vectors, float *dest) {
    timeFunction;

    for (int i = 0; i < vectors.n; i++) {
        dest[i] = oj__calculate_sqr_norm(vectors, i);
    }
}

internal void oj__get_clusters(
        Vectors subvectors,
        float *subvector_sqr_norms,
        Vectors centroids,
        float *centroid_sqr_norms,
        Vectors cluster_sums,
        int *cluster_amount, 
        float *distances) {
    timeFunction;

    // NOTE(fede): Instead of calculating the distance, needing to add the 
    //              square norms to each distance before the following loop, 
    //              we can integrate the square norm calculation here.
    oj__calculate_sqr_norms(centroids, centroid_sqr_norms);

    Vectors dotproduct_vectors = {
        .vecs = distances, 
        .n = subvectors.n,
        .dim = centroids.n,
    };
    oj__multiply_vectors_blas(subvectors, centroids, dotproduct_vectors, -2, 0);

    for (int i = 0; i < subvectors.n; i++) {
        float min_distance = distances[i * centroids.dim];
        min_distance += centroid_sqr_norms[0];

        int closest_centroid = 0;

        for (int j = 1; j < centroids.n; j++) {
            float distance = distances[i * centroids.n + j] 
                + centroid_sqr_norms[j];
            if (min_distance > distance) {
                closest_centroid = j; 
                min_distance = distance;
            } 
        }

        min_distance += subvector_sqr_norms[i];

        float *vec = oj__get_vector(subvectors, i);
        float *cluster_sum = oj__get_vector(
                cluster_sums,
                closest_centroid);

        oj__add_vectors(cluster_sum, vec, subvectors.dim, cluster_sum);
        cluster_amount[closest_centroid]++;
    }
}

internal void oj__update_centroids(
        Vectors subvectors,
        Vectors centroids,
        Vectors cluster_sums,
        int *cluster_amount,
        oj__bool *should_stop) {
    timeFunction;

    // TODO(fede): parametize this
    const float epsilon = 0.00001;
    *should_stop = oj__true;

    for (int i = 0; i < centroids.n; i++) {
        float *centroid = oj__get_vector(centroids, i);

        if (cluster_amount[i] > 0) {
            float *new_centroid = oj__get_vector(cluster_sums, i);

            oj__vector_div(
                    new_centroid,
                    subvectors.dim,
                    cluster_amount[i],
                    new_centroid);

            if (oj__vector_distance2(
                        centroid,
                        new_centroid,
                        subvectors.dim) > epsilon) {
                *should_stop = oj__false;
            }

            oj__vector_copy(new_centroid, centroid, subvectors.dim);
        } else {
            *should_stop = oj__false;
            float *new_centroid = oj__get_random_vector(subvectors);
            oj__vector_copy(new_centroid, centroid, subvectors.dim);
        }
    }
}

// NOTE(fede): 
//      cluster vector sums and cluster amount are pre_allocated 
internal void oj__get_kmeans_cluster_centroids(
        Vectors subvectors,
        Vectors centroids,
        int max_iterations,
        Vectors cluster_sums,
        int *cluster_amount,
        float *subvector_min_distances2) {
    assert(cluster_sums.dim == centroids.dim);
    assert(cluster_sums.n == centroids.n);

    srand(07734); // hello

    float *subvector_sqr_norms = oj__aligned_alloc(32, 
            sizeof(float) * subvectors.n);
    assert(subvector_sqr_norms);
    oj__calculate_sqr_norms(subvectors, subvector_sqr_norms);

    float *centroid_sqr_norms = oj__aligned_alloc(32,
            sizeof(float) * centroids.n);
    assert(centroid_sqr_norms);

    float *distances = oj__aligned_alloc(32,
            sizeof(float) * subvectors.n * centroids.n); 
    assert(distances);

    {
        timeBlock("Initialize centroids");

        oj__clear_array(cluster_amount, centroids.n, 0);
        oj__kmeans_plus_plus_init(
                centroids,
                subvectors,
                subvector_min_distances2,
                subvector_sqr_norms);
    }

    oj__bool should_stop = oj__false;

    {
        // timeBlock("Kmeans calculate centroids");
        timeBandwidth("Kmeans calculate centroids",
                subvectors.n * subvectors.dim * sizeof(float) + 
                centroids.n * centroids.dim * sizeof(float));
        for (int i = 0; !should_stop && i < max_iterations; i++) {
            oj__clear_array(cluster_sums.vecs,
                    cluster_sums.n * cluster_sums.dim,
                    0);
            oj__clear_array(cluster_amount, centroids.n, 0);

            oj__get_clusters(
                    subvectors,
                    subvector_sqr_norms,
                    centroids,
                    centroid_sqr_norms,
                    cluster_sums,
                    cluster_amount, 
                    distances);

            // oj__get_clusters_old(
            //         subvectors,
            //         centroids,
            //         cluster_sums,
            //         cluster_amount);

            oj__update_centroids(
                    subvectors,
                    centroids,
                    cluster_sums,
                    cluster_amount,
                    &should_stop);
        }
    }

    oj__aligned_free(subvector_sqr_norms);
    oj__aligned_free(centroid_sqr_norms);
    oj__aligned_free(distances);
}

void index_pq_train(IndexPQ *index, float *vectors, int n_vectors) {
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
        index->codebook = oj__aligned_alloc(32, codebook_size);

    assert(index->codebook);

    Vectors subvectors = {
        .vecs = oj__aligned_alloc(32,
                index->subvector_dimension *
                // index->n_subvectors * 
                n_vectors *
                sizeof(float)),
        .dim = index->subvector_dimension,
        .n = n_vectors,
    };
    assert(subvectors.vecs);

    float *prealloced_cluster_vector_sums = 
        malloc(index->centroids_per_page * subvectors.dim * sizeof(float));
    assert(prealloced_cluster_vector_sums);

    Vectors cluster_sums = {
        .vecs = prealloced_cluster_vector_sums,
        .dim = subvectors.dim,
        .n = index->centroids_per_page,
    };

    int *prealloced_cluster_amount = malloc(
            index->centroids_per_page * sizeof(int));
    assert(prealloced_cluster_amount);

    float *preallocated_subvector_distances2 = oj__aligned_alloc(32,
            subvectors.n * sizeof(float));
    assert(preallocated_subvector_distances2);

    for (int i = 0; i < index->n_subvectors; i++) {
        Vectors centroids = {
            .vecs = &index->codebook[
                i * index->centroids_per_page * index->subvector_dimension],
            .dim = index->subvector_dimension,
            .n = index->centroids_per_page,
        };

        {
            timeBandwidth(
                    "Copy subvectors to contiguous arrays",
                    n_vectors * index->subvector_dimension * sizeof(float));

            for (int j = 0; j < n_vectors; j++) {
                int global_subvector_index = 
                    j * index->dimension +
                    i * index->subvector_dimension;

                // STUDY(fede): check if memcpy is faster
                for (int vec_idx = 0;
                        vec_idx < index->subvector_dimension;
                        vec_idx++) {
                    subvectors.vecs[j * index->subvector_dimension + vec_idx] = 
                        vectors[global_subvector_index + vec_idx];
                }
            }
        }

        oj__get_kmeans_cluster_centroids(
                subvectors,
                centroids,
                index->n_iter,
                cluster_sums,
                prealloced_cluster_amount,
                preallocated_subvector_distances2);
    }

    free(prealloced_cluster_vector_sums);
    free(prealloced_cluster_amount);
    oj__aligned_free(preallocated_subvector_distances2);
}

void index_pq_add(
        IndexPQ *index,
        float *vectors,
        int n_vectors) {
    timeFunction;

    // TODO(fede): allow multiple adds
    assert(index->n_vectors == 0);
    index->n_vectors = n_vectors;

    if (!index->quantized_codes)
        index->quantized_codes = 
            malloc(sizeof(int) * index->n_subvectors * n_vectors);

    assert(index->quantized_codes);

    for (int i = 0; i < n_vectors; i++) {
        float *vector = &vectors[i * index->dimension];
        int *vector_codes = &index->quantized_codes[i * index->n_subvectors];

        for (int j = 0; j < index->n_subvectors; j++) {
            float *subvector = &vector[j * index->subvector_dimension];

            Vectors centroids = {
                .vecs = &index->codebook[
                    j * index->subvector_dimension * index->centroids_per_page],
                .dim = index->subvector_dimension,
                .n = index->centroids_per_page,
            };

            int code = oj__get_closest_centroid(
                    centroids,
                    subvector);

            vector_codes[j] = code;
        }
    }
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

IndexPQ_SearchResult index_pq_search(
        IndexPQ *index,
        float *vectors,
        int n_vectors,
        int n_neighbours) {
    timeFunction;
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
    assert(distance_lookup);
    oj__clear_array(
            distance_lookup,
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

    return result;
}

#endif // JUMPSUIT_IMPLEMENTATION

#ifdef __cplusplus
}
#endif

#endif // JUMPSUIT_H
