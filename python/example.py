import numpy as np
from orange_jumpsuit import IndexPQ

# Generate fake data
np.random.seed(42)
train_data = np.random.rand(1000, 64).astype(np.float32)
query_data = np.random.rand(5, 64).astype(np.float32)

# Use the library
index = IndexPQ(dimension=64, n_subvectors=8, n_bits_per_value=8)
index.train(train_data)
index.add(train_data)

distances, indices = index.search(query_data, n_neighbours=10)
print(indices[0][:3])
