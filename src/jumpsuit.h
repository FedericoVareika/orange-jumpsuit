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
void index_pq_train(IndexPQ *index, float *vectors, int n_vectors);

// TODO(fede): this will allocate memory, pass arena to it.
void index_pq_add(IndexPQ *index, float *vectors, int n_vectors);

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
    int n;
    int dim;
} Vectors;

internal float *oj__get_vector(Vectors vectors, int index) {
    return &vectors.vecs[vectors.dim * index];
}

internal float *oj__get_random_vector(Vectors vectors) {
    return oj__get_vector(vectors, oj__randi(0, vectors.n));
}

internal void oj__transpose_vectors(Vectors vectors, Vectors t_vectors) {
    for (int row = 0; row < vectors.n; row++) {
        for (int col = 0; col < vectors.dim; col++) {
            int pos = row * vectors.dim + col;

            int t_row = col;
            int t_col = row;
            int t_pos = t_row * t_vectors.dim + t_col;
            t_vectors.vecs[t_pos] = vectors.vecs[pos];
        }
    }
}

#include <immintrin.h>

// NOTE(fede): Pack b block into b_panel.
//  **b_panel should be able to fit entirely into L3 cache**
internal void oj__pack_b_panel(
        Vectors t_b, 
        int nc, int kc, int nr, 
        float *b_panel, int jc) {
    float *b_panel_macro_col = b_panel;
    int col = 0; 

    int micro_kernel_col = 0;
    while (col < nc) {

        // NOTE(fede): When all 'nr' cols have been filled, move to 
        //             the next b_panel_macro_col
        if (micro_kernel_col == nr) {
            micro_kernel_col = 0;
            b_panel_macro_col = &b_panel_macro_col[kc * nr]; 
        }

        float *col_vec = oj__get_vector(t_b, jc + col);

        for (int row = 0; row < kc; row++) {
            b_panel_macro_col[row * nr] = col_vec[row];
        } 

        col++;
        micro_kernel_col++;
    }
}

// NOTE(fede): Pack a block into a_block
//  **a_block should be able to fit entirely into L2 cache**
internal void oj__pack_a_block(
        Vectors a, 
        int mc, int kc, int mr,
        float *a_block, int ic) { 
    float *a_block_macro_row = a_block;
    int row = 0; 

    int micro_kernel_row = 0;
    while (row < mc) {

        // NOTE(fede): When all 'nr' cols have been filled, move to 
        //             the next b_panel_macro_col
        if (micro_kernel_row == mr) {
            micro_kernel_row = 0;
            a_block_macro_row = &a_block_macro_row[kc * mr]; 
        }

        float *row_vec = oj__get_vector(a, ic + row);

        for (int col = 0; col < kc; col++) {
            a_block_macro_row[col * mr] = row_vec[col];
        } 

        row++;
        micro_kernel_row++;
    }
}

internal inline void oj__multiply_micro_kernel(
        Vectors c, float *a_sliver, float *b_sliver, 
        int nr, int mr,
        int jc, int ic, int kc,
        int jr, int ir) {

    __m256 mB0; 
    __m256 mB1;
    __m256 mA0;
    __m256 mA1;

    __m256 result0_0  = _mm256_set1_ps(0); 
    __m256 result1_0  = _mm256_set1_ps(0);
    __m256 result2_0  = _mm256_set1_ps(0);
    __m256 result3_0  = _mm256_set1_ps(0);
    __m256 result4_0  = _mm256_set1_ps(0);
    __m256 result5_0  = _mm256_set1_ps(0);
    __m256 result0_1  = _mm256_set1_ps(0);
    __m256 result1_1  = _mm256_set1_ps(0);
    __m256 result2_1  = _mm256_set1_ps(0);
    __m256 result3_1  = _mm256_set1_ps(0);
    __m256 result4_1  = _mm256_set1_ps(0);
    __m256 result5_1  = _mm256_set1_ps(0);

	// This is the same for loop as in naive implementation, except now instead of the k indexing
	// a single dot product of 2 vectors of size k (a row of A and a col of B),
	// the k is indexing 6 rows of A and 16 cols of B
	// Since the SIMD width is 8 (256 bits), need to do 12 fmas here

    for(int k=0; k<kc; ++k) {

		// Load the k'th row of the B block 
        // (load twice since in total, it's 16 floats)
        mB0   = _mm256_load_ps(&b_sliver[k * nr + 8*0]);
        mB1   = _mm256_load_ps(&b_sliver[k * nr + 8*1]);

        // Load a single value for the k'th col of A
        // In total, we need to do this 6 times (col of A has height 6)
        // Note: the addresses below must be aligned on a 32-byte boundary
        mA0   = _mm256_set1_ps(a_sliver[0]);    // Load float @ A's col k, row m+0 into reg
        mA1   = _mm256_set1_ps(a_sliver[1]);    // Load float @ A's col k, row m+1
        // Now we have the 16 floats of B in mB0|mB1, and the 2 floats
        // of A broadcast in mA0 and mA1.
        result0_0      = _mm256_fmadd_ps(mB0,mA0,result0_0); // result = arg1 .* arg2 .+ arg3
        result0_1      = _mm256_fmadd_ps(mB1,mA0,result0_1);
        result1_0      = _mm256_fmadd_ps(mB0,mA1,result1_0);
        result1_1      = _mm256_fmadd_ps(mB1,mA1,result1_1);
        // result0_0 now contains the final result, for this k,
        // of row 0 and cols 0-7.
        // result0_1 now contains the final result, for this k,
        // of row 0 and cols 8-15.
        // result1_0 now contains the final result, for this k,
        // of row 1 and cols 0-7.
        // result1_1 now contains the final result, for this k,
        // of row 1 and cols 8-15.
        
        // Repeat for the other 4
        
        mA0   = _mm256_set1_ps(a_sliver[2]);
        mA1   = _mm256_set1_ps(a_sliver[3]);
        result2_0      = _mm256_fmadd_ps(mB0,mA0,result2_0);
        result2_1      = _mm256_fmadd_ps(mB1,mA0,result2_1);
        result3_0      = _mm256_fmadd_ps(mB0,mA1,result3_0);
        result3_1      = _mm256_fmadd_ps(mB1,mA1,result3_1);
        
        mA0   = _mm256_set1_ps(a_sliver[4]);
        mA1   = _mm256_set1_ps(a_sliver[5]);
        result4_0      = _mm256_fmadd_ps(mB0,mA0,result4_0);
        result4_1      = _mm256_fmadd_ps(mB1,mA0,result4_1);
        result5_0      = _mm256_fmadd_ps(mB0,mA1,result5_0);
        result5_1      = _mm256_fmadd_ps(mB1,mA1,result5_1);
    }
    
    float *c_row_0 = oj__get_vector(c, ic + ir + 0);
    float *c_row_1 = oj__get_vector(c, ic + ir + 1);
    float *c_row_2 = oj__get_vector(c, ic + ir + 2);
    float *c_row_3 = oj__get_vector(c, ic + ir + 3);
    float *c_row_4 = oj__get_vector(c, ic + ir + 4);
    float *c_row_5 = oj__get_vector(c, ic + ir + 5);

    // Write registers back to C
    *((__m256*) (&c_row_0[jc+jr+0*8])) += result0_0;
    *((__m256*) (&c_row_0[jc+jr+1*8])) += result0_1;
    *((__m256*) (&c_row_1[jc+jr+0*8])) += result1_0;
    *((__m256*) (&c_row_1[jc+jr+1*8])) += result1_1;
    *((__m256*) (&c_row_2[jc+jr+0*8])) += result2_0;
    *((__m256*) (&c_row_2[jc+jr+1*8])) += result2_1;
    *((__m256*) (&c_row_3[jc+jr+0*8])) += result3_0;
    *((__m256*) (&c_row_3[jc+jr+1*8])) += result3_1;
    *((__m256*) (&c_row_4[jc+jr+0*8])) += result4_0;
    *((__m256*) (&c_row_4[jc+jr+1*8])) += result4_1;
    *((__m256*) (&c_row_5[jc+jr+0*8])) += result5_0;
    *((__m256*) (&c_row_5[jc+jr+1*8])) += result5_1;
}

#define NC 1536
#define KC 240
#define MC 120
#define NR 16
#define MR 6

internal void oj__multiply_vectors_blocking(
        Vectors a, Vectors t_b, Vectors c, 
        int nc, int kc, int mc, int nr, int mr,
        float *b_panel, float *a_block) {
    timeFunction;
    assert(a.dim == t_b.dim);

    int n = t_b.n; 
    int m = a.n; 
    int k = a.dim; 

    assert(nc % nr == 0);
    assert(mc % mr == 0);

    // TODO(fede): include support for varying micro kernels, but since i found 
    //              an implementation of a 16x6 micro kernel multiply, i will 
    //              use it.
    assert(nr == 16);
    assert(mr == 6);

    for (int jc = 0; jc < n; jc += nc) {
        for (int pc = 0; pc < k; pc += kc) {

            oj__pack_b_panel(t_b, nc, kc, nr, b_panel, jc);

            for (int ic = 0; ic < m; ic += mc) {

                oj__pack_a_block(a, mc, kc, mr, a_block, ic);

                for (int jr = 0; jr < nc; jr += nr) {
                    float *b_sliver = &b_panel[jr * kc];

                    for (int ir = 0; ir < mc; ir += mr) {
                        float *a_sliver = &a_block[ir * kc];

                        oj__multiply_micro_kernel(
                                c, a_sliver, b_sliver,
                                nr, mr,
                                jc, ic, kc,
                                jr, ir);
                    }
                }
            }
        }
    }
} 

internal void oj__init_blocking_vectors(
        int nc, int kc, int mc,
        float **b_panel, float **a_block) {
    *b_panel = aligned_alloc(32, sizeof(float) * kc * nc);
    *a_block = aligned_alloc(32, sizeof(float) * mc * kc);
}

internal void oj__multiply_vectors_simd(Vectors a, Vectors t_b, Vectors c) {
    timeFunction;
    assert(a.dim == t_b.dim);

    // STUDY(fede): Change the order of the loops, simd may benefit knowing a 
    //              is larger
    for (int a_i = 0; a_i < a.n; a_i++) {
        float *a_vec = oj__get_vector(a, a_i);
        float *c_vec = oj__get_vector(c, a_i);
        for (int b_i = 0; b_i < t_b.n; b_i++) {
            c_vec[b_i] = 0;
            float *b_vec = oj__get_vector(t_b, b_i);

            __m256 result = _mm256_set1_ps(0);

            int k = 0;
            while (k < a.dim - 7) {

                __m256 mA = _mm256_load_ps(&a_vec[k]);
                __m256 mB = _mm256_load_ps(&b_vec[k]);

                result = _mm256_fmadd_ps(mA, mB, result);

                c_vec[b_i] += a_vec[k] * b_vec[k];
                k += 8;
            }

            float *farr = (float *)&result;
            c_vec[b_i] = 
                farr[0] + farr[1] + farr[2] + farr[3] +
                farr[4] + farr[5] + farr[6] + farr[7];

            while (k < a.dim) {
                c_vec[b_i] += a_vec[k] * b_vec[k];
                k++;
            }
        }
    }
}

#include <cblas.h>

internal void oj__multiply_vectors_blas(Vectors a, Vectors t_b, Vectors c) {
    timeFunction; 

    float *A = a.vecs;
    float *BT = t_b.vecs;
    float *C = c.vecs;

    int M = a.n; 
    int K = a.dim; 
    int N = t_b.n;

    // C = alpha*a*b + beta*c
    float alpha = 1;
    float beta = 0;

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

// TODO(fede): Naive implementation of matrix multiplication
internal void oj__multiply_vectors(Vectors a, Vectors t_b, Vectors c) {
    timeFunction;
    assert(a.dim == t_b.dim);
    for (int a_i = 0; a_i < a.n; a_i++) {
        float *a_vec = oj__get_vector(a, a_i);
        float *c_vec = oj__get_vector(c, a_i);
        for (int b_i = 0; b_i < t_b.n; b_i++) {
            c_vec[b_i] = 0;
            float *b_vec = oj__get_vector(t_b, b_i);
            for (int k = 0; k < a.dim; k++) {
                c_vec[b_i] += a_vec[k] * b_vec[k];
            }
        }
    }
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
// TODO(fede): this will allocate memory, pass arena to it.
internal void oj__kmeans_plus_plus_init(
        Vectors centroids,
        Vectors subvectors,
        float *min_distances2) {

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
        float sqr_norm = 0;
        for (int j = 0; j < vectors.dim; j++) {
            float v = vectors.vecs[i * vectors.dim + j];
            sqr_norm += v * v;
        }

        dest[i] = sqr_norm;
    }
}

float *A_BLOCK, *B_PANEL;
internal void oj__get_distances(
        Vectors subvectors, 
        float *subvector_sqr_norms,
        Vectors centroids, 
        float *centroid_sqr_norms,
        float *distances, 
        float *a_block, float *b_panel) {
    oj__calculate_sqr_norms(centroids, centroid_sqr_norms);

    // NOTE(fede): be careful, this is reusing the distances memory, so it 
    //             should not be overwritten.
    Vectors dotproduct_vectors = {
        .vecs = distances, 
        .n = subvectors.n,
        .dim = centroids.n,
    };
    

    // oj__multiply_vectors(subvectors, centroids, dotproduct_vectors);
    // oj__multiply_vectors_simd(subvectors, centroids, dotproduct_vectors);
    // oj__multiply_vectors_blocking(
    //         subvectors, centroids, dotproduct_vectors, 
    //         min(NC, centroids.n), KC, MC, NR, MR, b_panel, a_block);
    
    oj__multiply_vectors_blas(subvectors, centroids, dotproduct_vectors);

    {
        timeBlock("Final sum for distances");
        for (int i = 0; i < subvectors.n; i++) {
            for (int j = 0; j < centroids.n; j++) {
                int pos = i * centroids.n + j;
                float dotproduct = distances[pos]; 
                distances[pos] = 
                    subvector_sqr_norms[i]
                    + centroid_sqr_norms[j] 
                    - 2 * dotproduct;
            } 
        } 
    }
}

internal void oj__get_clusters_old(
        Vectors subvectors,
        Vectors centroids,
        Vectors cluster_sums,
        int *cluster_amount) {
    timeFunction;

    for (int vec_idx = 0; vec_idx < subvectors.n; vec_idx++) {
        float *vec = oj__get_vector(subvectors, vec_idx);

        int closest_centroid_idx = oj__get_closest_centroid(
                centroids, vec);

        float *cluster_sum = oj__get_vector(
                cluster_sums,
                closest_centroid_idx);

        oj__add_vectors(cluster_sum, vec, subvectors.dim, cluster_sum);
        cluster_amount[closest_centroid_idx]++;
    }
}

internal void oj__get_clusters(
        Vectors subvectors,
        float *subvector_sqr_norms,
        Vectors centroids,
        float *centroid_sqr_norms,
        Vectors cluster_sums,
        int *cluster_amount, 
        float *distances, 
        float *a_block, float *b_panel) {
    timeFunction;

    oj__get_distances(
            subvectors,
            subvector_sqr_norms,
            centroids,
            centroid_sqr_norms,
            distances, 
            a_block, a_block);

    for (int i = 0; i < subvectors.n; i++) {
        float *subvector = oj__get_vector(subvectors, i);

        float min_distance = distances[i * centroids.dim];
        int closest_centroid = 0;

        for (int j = 1; j < centroids.n; j++) {
            float distance = distances[i * centroids.n + j];
            if (min_distance > distance) {
                closest_centroid = j; 
                min_distance = distance;
            } 

#if 0
            float *centroid = oj__get_vector(centroids, j);
            float calculated_distance = oj__vector_distance2(
                        subvector,
                        centroid,
                        subvectors.dim);
            assert(distance + 0.005 >= calculated_distance 
                    && distance - 0.005 <= calculated_distance);
#endif
        }

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
        bool *should_stop) {
    timeFunction;

    // TODO(fede): parametize this
    const float epsilon = 0.00001;
    *should_stop = true;

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
                *should_stop = false;
            }

            oj__vector_copy(new_centroid, centroid, subvectors.dim);
        } else {
            *should_stop = false;
            float *new_centroid = oj__get_random_vector(subvectors);
            oj__vector_copy(new_centroid, centroid, subvectors.dim);
        }
    }
}

// TODO(fede): this will allocate memory, pass arena to it.
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

    float *subvector_sqr_norms = aligned_alloc(32, 
            sizeof(float) * subvectors.n);
    assert(subvector_sqr_norms);
    oj__calculate_sqr_norms(subvectors, subvector_sqr_norms);

    float *centroid_sqr_norms = aligned_alloc(32,
            sizeof(float) * centroids.n);
    assert(centroid_sqr_norms);

    float *distances = aligned_alloc(32,
            sizeof(float) * subvectors.n * centroids.n); 
    assert(distances);

    {
        timeBlock("Initialize centroids");

        oj__clear_array(cluster_amount, centroids.n, 0);
        int *centroid_init_idxs = cluster_amount;
        oj__kmeans_plus_plus_init(
                centroids,
                subvectors,
                subvector_min_distances2);
    }

    bool should_stop = false;

    {
        timeBlock("Kmeans calculate centroids");
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
                    distances, 
                    A_BLOCK, B_PANEL);

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

    free(subvector_sqr_norms);
    free(centroid_sqr_norms);
    free(distances);
}

// TODO(fede): this will allocate memory, pass arena to it.
void index_pq_train(IndexPQ *index, float *vectors, int n_vectors) {
    timeBandwidth(__func__, n_vectors * index->dimension * sizeof(float));

    oj__init_blocking_vectors(NC, KC, MC, &B_PANEL, &A_BLOCK);

    //            (C_i = [c_1, c_2, ..., c_k*])
    // codebook = [C_1, C_2, ..., C_m]
    
    int codebook_size =
            sizeof(float) *
            index->subvector_dimension *
            index->centroids_per_page * 
            index->n_subvectors;

    // NOTE(fede): Allow user to query length and allocate this themselves
    if (!index->codebook)
        index->codebook = aligned_alloc(32, codebook_size);

    assert(index->codebook);

    Vectors subvectors = {
        .vecs = aligned_alloc(32,
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

    float *preallocated_subvector_distances2 = malloc(
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
    free(preallocated_subvector_distances2);

    // free(B_PANEL);
    // free(A_BLOCK);
}

// TODO(fede): this will allocate memory, pass arena to it.
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

    return result;
}

#endif // JUMPSUIT_IMPLEMENTATION

#ifdef __cplusplus
}
#endif

#endif // JUMPSUIT_H
