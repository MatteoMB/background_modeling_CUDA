#!/bin/bash
JPEG=/home/user.dibris12/src/cpp/jpeg
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$JPEG/lib
rm -f a.out
rm -f ../../data/PETS2006/in*_*
nvcc -std=c++11 -w -Wno-deprecated-gpu-targets -I$JPEG/include -L$JPEG/lib -ljpeg -O2 -lm -lpthread -lX11 hpc_cuda.cu
./a.out ../../data/PETS2006/
