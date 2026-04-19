#include <Python.h>

/* Define the empty module */
static struct PyModuleDef dummy_module = {
    PyModuleDef_HEAD_INIT,
    "_dummy", /* name of module */
    NULL,     /* module documentation */
    -1,       /* size of per-interpreter state */
    NULL      /* methods */
};

/* The initialization function the linker is looking for */
PyMODINIT_FUNC PyInit__dummy(void) {
    return PyModule_Create(&dummy_module);
}
