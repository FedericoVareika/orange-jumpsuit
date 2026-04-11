#!/bin/sh

common_flags_internal="-ffile-prefix-map=old=new -g -W -Og"
common_flags_external="-O3 -ffast-math"
linker_flags="-lm -mavx2 -mfma"

mkdir -p build

gcc -std=gnu11 \
    $linker_flags \
    $common_flags_external \
    src/jumpsuit.c  \
    -shared \
    -o build/libjumpsuit.so 
