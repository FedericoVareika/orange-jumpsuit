#!/bin/sh

common_flags_internal="-ffile-prefix-map=old=new -g -W -Og"
common_flags_external="-O3 -ffast-math -Wall -m64"
linker_flags="-lm -lopenblas"

mkdir -p build/tests

CC=${CC:-gcc}

$CC -std=gnu11 \
    $linker_flags \
    $common_flags_external \
    src/tests/linux_jumpsuit.c  \
    -D'PROFILER=1' \
    -o build/tests/orange-jumpsuit 
