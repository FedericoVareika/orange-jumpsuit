#!/bin/sh

common_flags_internal="-ffile-prefix-map=old=new -g -W -Og"
common_flags_external="-O3"
linker_flags="-lm"

mkdir -p build

gcc -std=gnu11 \
    $linker_flags \
    $common_flags_internal \
    -D'PROFILER=0' \
    src/linux_jumpsuit.c  \
    -o build/orange-jumpsuit 
    # $common_flags_external \
