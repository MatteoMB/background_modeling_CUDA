#!/bin/bash
JPEG=/home/user.dibris12/src/cpp/jpeg
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$JPEG/lib
rm -f a.out
rm -f ../../data/PETS2006/in*_*
g++ -std=c++11 -fopenmp -I$JPEG/include -L$JPEG/lib -ljpeg -O2 -lm -lpthread -lX11 sequential.cpp
./a.out ../../data/PETS2006/
