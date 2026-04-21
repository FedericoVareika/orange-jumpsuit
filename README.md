## Indice de Quantizacion de Producto (PQ) para busqueda vectorial 

Esta libreria provee busqueda vectorial con PQ en C, con un wrapper de Python. 

## API

### C 

Funciones expuestas: 

```c
IndexPQ index_pq_init(int dimension, int n_subvectors, int n_bits_per_value);
void index_pq_train(IndexPQ *index, float *vectors, int n_vectors);
void index_pq_add(IndexPQ *index, float *vectors, int n_vectors);
IndexPQ_SearchResult index_pq_search(IndexPQ *index, float *vectors, int n_vectors, int n_neighbours);
```

Para usar la API en c, solo hay que descargar `jumpsuit.h` e incluirlo. 

#### Tests

El codigo de prueba en `jumpsuit/tests` solo funciona para Linux. 
Para compilar el codigo de prueba, se debe correr `./build_tests.sh`, y el 
ejecutable queda en `./build/tests/orange-jumpsuit`.

### Python

Codigo de uso: 

```python
from orange_jumpsuit import IndexPQ

index = IndexPQ(dimension, n_subvectors, n_bits_per_value)
index.train(train_data)
index.add(train_data)
distances, indices = index.search(query_data, n_neighbours)
```

Para instalar la libreria en python se puede usar pip: 

```
pip install orange-jumpsuit
```

Esta libreria solo esta para Linux y Windows, y es posible que se intente de 
compilar si no existe el codigo pre-compilado en 
[PyPI](https://pypi.org/project/orange-jumpsuit/). Si esto ocurre, el proyecto 
se intenta de compilar con `gcc` (Linux) o `cl.exe` (MSVC en Windows).

Para armarla localmente puede correr `build_lib.(sh/bat)` para 
obtener `libjumpsuit.(so/dll)`, o `build_python_wheel.(sh/bat)` para obtener 
`dist/(...).whl` que puede incluir en python con `pip install [camino a .whl]`.

Estos scripts son para Linux y Windows respectivamente.
