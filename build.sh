#!/bin/sh

common_flags_internal="-ffile-prefix-map=old=new -g -W"
common_flags_external="-O3 -ffast-math -W -m64"
linker_flags="-lm -mavx2 -mfma -lopenblas"

mkdir -p build

gcc -std=gnu11 \
    $linker_flags \
    $common_flags_external \
    src/linux_jumpsuit.c  \
    -o build/orange-jumpsuit 
    # -D'PROFILER=0' \
    # $common_flags_internal \

# gcc -std=gnu11 \
#     $linker_flags \
#     $common_flags_internal \
#     src/linux_jumpsuit.c  \
#     -o build/orange-jumpsuit 
#     # -D'PROFILER=0' \
#     # $common_flags_internal \
