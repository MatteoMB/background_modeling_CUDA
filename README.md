#HPC Project: background modeling with cuda


- FILES EXPLANATION
    - Source code
        1) sequential.cpp         c++ sequential implementation of the algorithm
        2) hpc_cuda.cu            cuda implementation with managed memory
        3) mp_cuda                cuda implementation with I/O parallelization with openmp
        4) hpc_cuda_no_managed.cu cuda implementation without managed memory
    - Performance measurements
        seq_total.txt        time per frame of 1)
        cuda_total.txt       time per frame of 2)
        mp.txt               time per frame of 3)
        load.txt             time for loading an input image in memory
        store.txt            time for storing an image on disk
        seq_processing.txt   time for processing an image (no I/O) with 1)
        cuda_processing.txt  time for processing an image (no I/O) with 2)
        nvprof.txt           output of the cuda nvprofile program
        rgb_to_gray.txt      time for converting a image in grayscale (per frame)
        cuda_kernel.txt      time only for cuda kernel (per frame) 4)
        seq_kernel.txt       time only for sequential kernel function (per frame)
        seq_8x8_kernel.txt   time only for sequential kernel function (per frame) with 8x8 pixel patches
    - Libraries
        CImg.hpc_cuda             load and store images
        libjpeg                   manage .jpg files
    - Image Files
        some input and output to show what's going on.
        
- NOTES
    Short explanation
        In hpc_cuda.cu file the code is fully commented, to understand how it works.
        The aim of the program is to build a robust description of the background in order to segment
        moving elements. The program takes a directory name as input and processes all the jpg images inside it.
        In order to achieve this goal, each image is subdivided in patches and all the calculations are made locally.
        The representation of each patch is its histogram, the program compares histograms of the same patch in time.
        The basic idea for parallelization is that pixel values go from 0 to 256, so with 256 threads it's immediate
        to measure similarity between two of them. Moreover, the patch dimension are 16x16, so each thread takes
        care of 1 pixel of the image.
        The history of the patch is stored in a Dictionary of histograms, that updates when many anomalies are
        detected in a row in the same patch. When this happens, the histograms of the anomal patches are averaged
        and putted in the Dictionary, removing the least used histogram. In this way it's possible to achieve
        robustness.
    Comments on algorithm
        The patch descriptor (histogram) is very simple, so there is not much calculation to do with it.
        The output looks good, although shadows have strong impact on it, and there is some salt and pepper noise.
        The worst cases are where there is not much contrast between objects. However the most important things
        that are happening on the scene are detected (pedestrians walking). It's somewhat difficult for this implementation
        to survive to background changes, that are not recognized in stable way: maybe the average is not the best thing to do.
        Although the dictionary does seldom fill completely for the input given, maybe we can also change the
        metrics for histogram replacement. Anyway the updates are carried out: if we look at img001165 we see that
        the highlighted patch on the backpack that appears 75 frames before has disappeared (as it happened for the
        rest of the backpack).
        Parameters are pre-defined but can be changed; however the patch reduction is working only for the sequential
        program (8x8).
    Comments on performances
        Since the algorithm is very simple, the performances are quite good: the bottleneck in the whole program
        is the store phase in hard drive, so there is not much difference between the cuda and the sequential implementation.
        However if we compare the processing time of the two implementations, we notice that there is almost 2* speedup using cuda.
        Moreover we also notice that the slowest part of the computation is in common between sequential and cuda:
        the transformation of the image in grayscale, as we can see in rgb_to_gray.txt and in cuda_only.txt.
        The kernel function is performed in 0.2 ms by cuda (explicit data transfers included) and in 2.2 ms in the sequential.
        In order to overcome the I/O bottleneck, I used OpenMP to pipeline the jobs using two threads, and I managed to
        reach a great result: the time per frame is similar to the store time. Even using more threads the performances
        don't get better, perhaps since I reached the maximum I/O speed.
        In the multithreaded version, I used a busy wait approach in order to ensure data dependecies.
        The sequential version isn't optimized in any way.
    Dataset
        The dataset consist in 1200 sequential jpg images of 720x576 pixel in 3 channels rgb, taken from
        www.changedetection.net website (dataset PETS2006).
    Platform
        Performances are measured using a node Dual CPU-Intel E52695 V3 each with 28 core and 64GB RAM.
        Infiniband FDR 56Gb/s
        NVIDIA k40M GPU
