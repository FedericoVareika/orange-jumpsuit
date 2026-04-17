#!/bin/sh

perf record -F 99 -g -- ./build/orange-jumpsuit

perf script > perf/out.perf
pushd perf

perl ../../FlameGraph/stackcollapse-perf.pl out.perf > out.folded
perl ../../FlameGraph/flamegraph.pl --flamechart out.folded > profile.svg

popd
