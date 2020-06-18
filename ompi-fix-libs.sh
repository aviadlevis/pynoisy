#!/bin/sh
# https://github.com/open-mpi/ompi/issues/3705
prefix="/usr/lib/x86_64-linux-gnu/openmpi"
for filename in $(ls $prefix/lib/openmpi/*.so); do
    /home/aviad/anaconda3/envs/eht/bin/patchelf --add-needed libmpi.so.20 $filename
    /home/aviad/anaconda3/envs/eht/bin/patchelf --set-rpath "\$ORIGIN/.." $filename
done
