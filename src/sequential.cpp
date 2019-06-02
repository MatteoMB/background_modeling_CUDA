#include <glob.h>
#include <string.h>
#include <iostream>
#include <omp.h>
#define cimg_use_jpeg
#include "CImg.h"
using namespace cimg_library;
using namespace std;
#define RIDX 2
#define RIDY 2
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

typedef struct Dict {
    unsigned int D[D_DIM][D_HIST];
    unsigned int C[D_DIM];
    unsigned int A[A_DUR][D_HIST];
    unsigned int iHist = 0;
    unsigned int lenD = 0;
    unsigned int lenA = 0;
} Dict_t;
 
void kernel(unsigned char * frame, Dict_t * Dictionary, int h, int w){
    int blockx, blocky, x, y, l, i, j , k, b, avg, min, iMin, i_c, iHist, lenD, lenA, sim, iD;
    for (blocky = 0; blocky < h; blocky += PATCHY)
        for (blockx = 0; blockx < w; blockx += PATCHX){
            //dictionary index
            l = blocky / PATCHY * w / PATCHX + blockx / PATCHX;
            iHist = Dictionary[l].iHist;
            lenD = Dictionary[l].lenD;
            lenA = Dictionary[l].lenA;
            sim = 0;
            bool anomaly = 1;
            //b stands for bin
            for (b = 0; b < D_HIST; b ++)
                Dictionary[l].D[iHist][b] = 0;
            for (y = blocky; y < blocky + PATCHY; y++)
                for (x = blockx; x < blockx + PATCHX; x++)
                    Dictionary[l].D[iHist][frame[y * w + x] / (RIDX * RIDY)]++;
            if (iHist < INIT){
                Dictionary[l].iHist++;
                Dictionary[l].lenD++;
                continue;
            }
            for (iD = 0; iD < lenD; iD++){
                if (iD != iHist){
                    sim = 0;
                    for (b = 0; b < D_HIST; b++)
                        sim += Dictionary[l].D[iHist][b] < Dictionary[l].D[iD][b] ? Dictionary[l].D[iHist][b] : Dictionary[l].D[iD][b];
                    if (sim > SIM){
                        anomaly = 0;
                        Dictionary[l].C[iD]++;
                    }
                }
            }
            if(anomaly){
                for (y = blocky; y < blocky + PATCHY; y++)
                    for (x = blockx; x < blockx + PATCHX; x++)
                        frame[y * w + x] = frame[y * w + x]/5;
                for (b = 0; b < D_HIST; b++)
                    Dictionary[l].A[lenA][b] = Dictionary[l].D[iHist][b];
                if (lenA == A_DUR - 1){
                    for (b = 0; b < D_HIST; b++){
                        avg = 0;
                        for (int i_a = 0; i_a <= lenA; i_a++)
                            avg += Dictionary[l].A[i_a][b];
                        Dictionary[l].D[iHist][b] = avg ? avg / A_DUR + 1 : 0;
                        Dictionary[l].C[iHist] = 0;
                    }
                    if(lenD == D_DIM-1){
                        for (i_c = 0, min = 0, iMin = -1; i_c < lenD; i_c++){
                            if (i_c != iHist && (min > Dictionary[l].C[i_c] || iMin < 0)){
                                min = Dictionary[l].C[i_c];
                                iMin = i_c;
                            }
                        }
                        Dictionary[l].iHist = iMin;
                    }
                    else{
                        Dictionary[l].iHist++;
                        Dictionary[l].lenD++;
                    }
                    Dictionary[l].lenA = 0;
                }
                else Dictionary[l].lenA++;
            }
            else Dictionary[l].lenA = 0;
        }
}

int main(int argc, char * argv[]){
    CImg<unsigned char> img;
    CImg<unsigned char> outImg;
    unsigned char * frame, * dFrame;
    Dict_t * Dictionary;
    glob_t globbuf;
    int h,w,i,j,iFrame,nFrames;

    if(argc<2){
        cout<<"Missing directory name"<<endl;
        return 1;
    }
    char *dir = strcat(argv[1], "*.jpg");
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
            dFrame = (unsigned char *)malloc(sizeof(unsigned char) * h * w);
            Dictionary = (Dict_t *) malloc(sizeof(Dict_t) * w / PATCHX * h / PATCHY);
        }
        if(img.height() != h || img.width()!=w){
            cerr<<"Frame dimensions don't match"<<endl;
            return 1;
        }

        for (i = 0; i < h; i++)
            for (j = 0; j < w; j++)
                dFrame[i * w + j] = (unsigned char)(0.299 * img(j, i, 0, 0) + 0.587 * img(j, i, 0, 1) + 0.114 * img(j, i, 0, 2));
#ifdef PERFORMANCE
        t_start = chrono::high_resolution_clock::now();
#endif
        kernel(dFrame, Dictionary ,h, w);
#ifdef PERFORMANCE
        t_end = chrono::high_resolution_clock::now();
        exec_time = t_end - t_start;
        cout << iFrame << " " << exec_time.count() * 1e3 << endl;
        perf[iFrame] = exec_time.count() * 1e3;
#endif
        outImg.assign(dFrame,w,h,1,1,true);

        outImg.save(globbuf.gl_pathv[iFrame],1,3);

#ifdef STOP
        if (iFrame > STOP)
        {
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

    free(dFrame);

    if (globbuf.gl_pathc > 0)
        globfree(&globbuf);
    return 0;

}