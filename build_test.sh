#!/bin/sh

common_flags_internal="-ffile-prefix-map=old=new -g -W -Og"
common_flags_external="-O3 -ffast-math -W -m64"
linker_flags="-lm -mavx2 -mfma -lopenblas" # -lopenblas -fopenmp

mkdir -p build

CC=${CC:-gcc}

$CC -std=gnu11 \
    $linker_flags \
    $common_flags_external \
    src/tests/linux_jumpsuit.c  \
    -D'PROFILER=1' \
    -o build/orange-jumpsuit 
    # $common_flags_internal \

# $CC -std=gnu11 \
#     $linker_flags \
#     $common_flags_internal \
#     src/linux_jumpsuit.c  \
#     -o build/orange-jumpsuit 
#     # -D'PROFILER=0' \
#     # $common_flags_internal \
