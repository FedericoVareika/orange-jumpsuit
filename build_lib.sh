#!/bin/sh

if echo "$CC" | grep -q "mingw"; then
    EXTENSION="dll"
else
    EXTENSION="so"
fi

common_flags_external="-O3 -ffast-math -W -m64"
linker_flags="-lm -mavx2 -mfma -lopenblas" # -lopenblas -fopenmp

mkdir -p build/lib

CC=${CC:-gcc}

echo "Compiling with $CC..."

$CC -std=gnu11 \
    $common_flags_external \
    -mavx2 -mfma \
    src/jumpsuit.c \
    -shared \
    -D'PROFILER=0' \
    -o build/lib/libjumpsuit.$EXTENSION $linker_flags

