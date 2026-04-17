import ctypes
import numpy as np
import os
import platform

# 1. Define the C structs in Python
class C_IndexPQ(ctypes.Structure):
    _fields_ = [
        ("dimension", ctypes.c_int),
        ("n_subvectors", ctypes.c_int),
        ("n_bits_per_value", ctypes.c_int),
        ("n_iter", ctypes.c_int),
        ("centroids_per_page", ctypes.c_int),
        ("subvector_dimension", ctypes.c_int),
        ("n_vectors", ctypes.c_int),
        ("codebook", ctypes.POINTER(ctypes.c_float)),
        ("quantized_codes", ctypes.POINTER(ctypes.c_int)),
    ]

class C_IndexPQ_SearchResult(ctypes.Structure):
    _fields_ = [
        ("n_vectors", ctypes.c_int),
        ("n_neighbours", ctypes.c_int),
        ("indices", ctypes.POINTER(ctypes.c_int)),
        ("distances", ctypes.POINTER(ctypes.c_float)),
    ]


# 2. Load the shared library
# Adjust the path/extension depending on your OS (e.g., .dll for Windows)
# lib_path = os.path.join(os.path.dirname(__file__), "libjumpsuit.so")
# jumpsuit_lib = ctypes.CDLL(lib_path)

# 2. Load the shared library
# Determine the library name based on the OS
system = platform.system()
libs_path = os.path.join(os.path.dirname(__file__), "libs")
if system == "Windows":
    lib_name = "libjumpsuit.dll"
    os.add_dll_directory(libs_path)
elif system == "Darwin":  # macOS
    lib_name = "libjumpsuit.dylib"
else:  # Linux and others
    lib_name = "libjumpsuit.so"

lib_path = os.path.join(os.path.dirname(__file__), lib_name)

try:
    # Use CDLL for standard C calling conventions (MinGW/GCC)
    jumpsuit_lib = ctypes.CDLL(lib_path)
except OSError as e:
    raise OSError(f"Could not load the shared library at {lib_path}. "
                  f"Ensure it is compiled for {system}.") from e

# 3. Define function signatures
# IndexPQ index_pq_init(int dimension, int n_subvectors, int n_bits_per_value);
jumpsuit_lib.index_pq_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
jumpsuit_lib.index_pq_init.restype = C_IndexPQ

# void index_pq_train(IndexPQ *index, float *vectors, int n_vectors);
jumpsuit_lib.index_pq_train.argtypes = [ctypes.POINTER(C_IndexPQ), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
jumpsuit_lib.index_pq_train.restype = None

# void index_pq_add(IndexPQ *index, float *vectors, int n_vectors);
jumpsuit_lib.index_pq_add.argtypes = [ctypes.POINTER(C_IndexPQ), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
jumpsuit_lib.index_pq_add.restype = None

# IndexPQ_SearchResult index_pq_search(IndexPQ *index, float *vectors, int n_vectors, int n_neighbours);
jumpsuit_lib.index_pq_search.argtypes = [
    ctypes.POINTER(C_IndexPQ), 
    ctypes.POINTER(ctypes.c_float), 
    ctypes.c_int, 
    ctypes.c_int
]
jumpsuit_lib.index_pq_search.restype = C_IndexPQ_SearchResult

# 4. Create the Pythonic wrapper
class IndexPQ:
    def __init__(self, dimension: int, n_subvectors: int, n_bits_per_value: int):
        """Initializes the Product Quantization Index."""
        self._c_index = jumpsuit_lib.index_pq_init(dimension, n_subvectors, n_bits_per_value)
        self.dimension = dimension

    def _prepare_vectors(self, vectors: np.ndarray):
        """Helper to ensure numpy arrays are 2D, contiguous, and float32."""
        vectors = np.atleast_2d(vectors)
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of dimension {self.dimension}, got {vectors.shape[1]}")
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def train(self, vectors: np.ndarray):
        """Trains the index on the provided vectors."""
        vecs_c = self._prepare_vectors(vectors)
        n_vectors = vecs_c.shape[0]
        
        # Pass by reference using ctypes.byref
        vecs_ptr = vecs_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        jumpsuit_lib.index_pq_train(ctypes.byref(self._c_index), vecs_ptr, n_vectors)

    def add(self, vectors: np.ndarray):
        """Adds vectors to the quantized index."""
        vecs_c = self._prepare_vectors(vectors)
        n_vectors = vecs_c.shape[0]
        
        vecs_ptr = vecs_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        jumpsuit_lib.index_pq_add(ctypes.byref(self._c_index), vecs_ptr, n_vectors)

    def search(self, vectors: np.ndarray, n_neighbours: int):
        """Searches the index. Returns (distances, indices)."""
        vecs_c = self._prepare_vectors(vectors)
        n_vectors = vecs_c.shape[0]
        
        vecs_ptr = vecs_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Call the C function
        result = jumpsuit_lib.index_pq_search(ctypes.byref(self._c_index), vecs_ptr, n_vectors, n_neighbours)
        
        # Convert the C pointers back to usable NumPy arrays
        total_elements = n_vectors * n_neighbours
        
        # Extract data from pointers and reshape to (n_queries, n_neighbours)
        distances = np.ctypeslib.as_array(result.distances, shape=(total_elements,)).reshape(n_vectors, n_neighbours).copy()
        indices = np.ctypeslib.as_array(result.indices, shape=(total_elements,)).reshape(n_vectors, n_neighbours).copy()
        
        # Note: Be careful with memory leaks here! (See note below)
        return distances, indices
