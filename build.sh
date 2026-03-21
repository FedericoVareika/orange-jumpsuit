#!/bin/sh

common_flags_internal="-ffile-prefix-map=old=new -g -W -Og"
linker_flags="-lm"

mkdir -p build

gcc -std=gnu11 \
    $linker_flags \
    $common_flags_internal \
    src/linux_jumpsuit.c  \
    -o build/orange-jumpsuit 
