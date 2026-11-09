#!/usr/bin/env bash
# Source from a shell, or let the Python launch/build helpers apply these paths.
_cluster_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -d "${_cluster_root}/deps/mpich/bin" ]]; then
  export PATH="${_cluster_root}/deps/mpich/bin:${PATH}"
  export LD_LIBRARY_PATH="${_cluster_root}/deps/mpich/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
unset _cluster_root
