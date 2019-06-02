#include <glob.h>
#include <string.h>
#include <iostream>
#include <omp.h>
#define cimg_use_jpeg
#include "CImg.h"
using namespace cimg_library;
using namespace std;
#define RIDX 1
#define RIDY 1
#define D_DIM 8
// #define STOP 30
#define D_HIST 256/(RIDX * RIDY)
#define PATCHX 16 / RIDX
#define PATCHY 16 / RIDY
#define INIT 4
#define SIM 135 / (RIDX * RIDY)
#define A_DUR 75
#define PERFORMANCE
#ifdef PERFORMANCE
    #include <ctime>
    #include <chrono>
#endif
//struct containing dictionary
__global__
typedef struct Dict {
    //dictionary: array of histograms, D_DIM represent how many background
    //models we want to store
    unsigned int D[D_DIM][D_HIST];
    //counter for the usage of the dictionary
    unsigned int C[D_DIM];
    //container of the histograms of anomalies that last in time (specified in
    //terms of A_DUR images). If an anomaly lasts more than A_DUR images, it's
    //added to the Dictionary (it represents a backgroud)
    unsigned int A[A_DUR][D_HIST];
    //indexes for the current histogram the current filling of D and A
    unsigned int iHist = 0;
    unsigned int lenD = 0;
    unsigned int lenA = 0;
} Dict_t;


__global__ 
void kernel(unsigned char * frame, Dict_t * Dictionary, int h, int w){
    //global indexes
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    //local index
    unsigned char k = threadIdx.y * blockDim.x + threadIdx.x;
    //dictionary index
    int l = w / blockDim.x * blockIdx.y + blockIdx.x;
    //iD is the cycling index of the Dictionary structure, sim stands for similarity
    //anomaly means if it's been detected an anomaly in the current patch
    //all the threads in the group must agree on these three variables
    __shared__ unsigned int iD, sim;
    __shared__ bool anomaly;
    if(!k){
        sim = 0;
        iD = 0;
        //by default we assume there's an anomaly
        anomaly = 1;
    }
    //p is the current pixel value
    //each thread works on one pixel (one patch => one thread group)
    unsigned char p = frame[i * w + j];
    int iHist = Dictionary[l].iHist;
    int lenD =  Dictionary[l].lenD;
    int lenA = Dictionary[l].lenA;
    int value, i_a, i_c, min, iMin, avg;
    //here we parallelize the initialization of the current histogram
    //(stored by default in the dictionary at index iHist) 
    Dictionary[l].D[iHist][k] = 0;
    __syncthreads();
    //building the histogram means only incrementing by 1 the corresponding bin
    atomicInc(&Dictionary[l].D[iHist][p],257);
    __syncthreads();
    //for the first INIT images, we just add their histograms to the Dictionary
    if(iHist < INIT){
        if(!k){
            Dictionary[l].C[iHist] = 0;
            Dictionary[l].iHist++;
            Dictionary[l].lenD++;
        }
        return;
    }
    //loop inside the dictionary, we want to compare the current histogram
    //with all the others in the dictionary
    for(; iD < lenD;){
        //The current histogram can be anywhere in the dictionary, we don't compare the histogram with itself
        if(iD != iHist){
            value = Dictionary[l].D[iHist][k];
            //the similarity is the intersection (minimum) between the corresponding bins of the two histograms
            atomicAdd(&sim,value < Dictionary[l].D[iD][k] ? value : Dictionary[l].D[iD][k]);
            __syncthreads();
            //if they are enough similar (>50%), all threads agree that there isn't an anomaly
            if(sim > SIM){
                anomaly = 0;
                //thread 0 register what histogram was used to reconstruct the patch
                if(!k) Dictionary[l].C[iD]++;
                break;
            } 
        }
        //iD++
        if(!k){
            sim = 0;
            atomicInc(&iD, D_DIM + 1);
        }
        __syncthreads();
    }
    if(anomaly){
        //each thread darkens its own pixel
        frame[i*w +j]= p/5;
        //copy current histogram in the container of the anomalies
        Dictionary[l].A[lenA][k] = value;
        __syncthreads();
        //the container of the anomalies is full: there is a change in the background, so we
        //add the average of the histograms of A to the dictionary. It will represent the new background
        if (lenA == A_DUR - 1){
            //each thread averages one bin
            for( i_a = 0, avg = 0; i_a < A_DUR; i_a ++)
                avg += Dictionary[l].A[i_a][k];
            //the average is rounded by excess since we take the minimum in the comparison
            Dictionary[l].D[iHist][k] = avg ? avg / A_DUR + 1 : 0;
            //initialize the Counter
            Dictionary[l].C[iHist] = 0;
                //the aim of this part is to substitute the least used histogram with the new one
                //(since the dictionary is full). It's difficult to parallelize, but the dictionary is
                //small (and almost never filled), so it's done sequentially
            if(!k){
                if(lenD == D_DIM-1){
                    //take the index of the minimum
                    for (i_c = 0, min = 0, iMin = -1; i_c < lenD; i_c++)
                        if(i_c != iHist && (min > Dictionary[l].C[i_c] || iMin < 0)){
                            min = Dictionary[l].C[i_c];
                            iMin = i_c;
                        }
                    //in the next loop,, the iMin-th histogram will be overwritten
                    Dictionary[l].iHist = iMin;
                }
                //there is enough space in the dictionary
                else{
                    Dictionary[l].iHist++;
                    Dictionary[l].lenD++;
                }
                //empty A
                Dictionary[l].lenA = 0;
            }
        }
        //we can take at least one more anomaly
        else if(!k) Dictionary[l].lenA++;
    }
    //no anomaly, background isn't changing: empty A
    else if(!k) Dictionary[l].lenA = 0;
}

int main(int argc, char * argv[]){
    CImg<unsigned char> img;
    CImg<unsigned char> outImg;
    unsigned char * dFrame, * frame;
    Dict_t * Dictionary;
    glob_t globbuf;
    int h,w,i,j,iFrame,nFrames;
    if(argc<2){
        cout<<"Missing directory name"<<endl;
        return 1;
    }
    char *dir = strcat(argv[1], "*.jpg");
    //take all jpg file in the specified directory as input
    if (glob (dir, GLOB_TILDE, NULL, &globbuf) != 0){
            cout << "Can't open the chosen directory" << endl;
            return 1;
    }
    nFrames = globbuf.gl_pathc;
    #ifdef PERFORMANCE
        chrono::high_resolution_clock::time_point t_start,t_end;
        chrono::duration<double> exec_time;
        double * perf = new double [nFrames];
    #endif
    for (iFrame = 0; iFrame < nFrames; iFrame++){
        img.load(globbuf.gl_pathv[iFrame]);
        if (iFrame == 0){
            h = img.height();
            w = img.width();
            //malloc the device Frame and the dictionary: data migration between host and GPU is managed by cuda
            frame = (unsigned char *) malloc(sizeof(unsigned char) * h * w);
            cudaMalloc(&dFrame,sizeof(unsigned char) * h * w);
            cudaMallocManaged(&Dictionary,sizeof(Dict_t) * w/PATCHX * h/PATCHY);
        }
        if(img.height() != h || img.width()!=w){
            cout<<"Frame dimensions don't match"<<endl;
            return 1;
        }
        //convert the image in grayscale and putting it the device array
        for (i = 0; i < h; i++)
            for (j = 0; j < w; j++)
                frame[i * w + j] = (unsigned char)(0.299 * img(j, i, 0, 0) + 0.587 * img(j, i, 0, 1) + 0.114 * img(j, i, 0, 2));
        #ifdef PERFORMANCE
            t_start = chrono::high_resolution_clock::now();
        #endif
        cudaMemcpy(dFrame, frame, sizeof(unsigned char) * h * w, cudaMemcpyHostToDevice);
        dim3 threadsPerBlock(PATCHX,PATCHY);
        dim3 numBlocks(w / threadsPerBlock.x, h / threadsPerBlock.y);
        kernel<<<numBlocks, threadsPerBlock>>>(dFrame, Dictionary ,h, w);
        cudaDeviceSynchronize();
        cudaMemcpy(frame, dFrame, sizeof(unsigned char) * h * w, cudaMemcpyHostToDevice);
        #ifdef PERFORMANCE
            t_end = chrono::high_resolution_clock::now();
            exec_time = t_end - t_start;
            cout <<iFrame<<" "<< exec_time.count() * 1e3<<endl;
            perf[iFrame] = exec_time.count() * 1e3;
        #endif
        //save the output
        for (i = 0; i < h; i++)
            for (j = 0; j < w; j++)
                outImg(j,i,0,0) = frame[i * w + j];

        outImg.save(globbuf.gl_pathv[iFrame],1,3);
        

        #ifdef STOP
            if(iFrame>STOP){ 
                nFrames = STOP + 1;
                break;
            }
        #endif
    }
        
    #ifdef PERFORMANCE
        double avg = 0;
        double stdev = 0;
        for(int iP = 0; iP < nFrames; iP++)
            avg += perf[iP];
        avg /= nFrames;
        for(int iP = 0; iP < nFrames; iP++)
            stdev += abs(perf[iP] - avg);
        stdev /= nFrames;
        cout<<"Avg time: "<<avg<<" ms"<<endl<<"STDev: "<<stdev<<" ms"<<endl;
    #endif

    // free
 
    cudaFree(dFrame);
    cudaFree(Dictionary);

    if (globbuf.gl_pathc > 0)
        globfree(&globbuf);
    return 0;

}