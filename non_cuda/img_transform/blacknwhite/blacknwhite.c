#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>

typedef double FLOAT;
//typedef float FLOAT;

typedef unsigned int UINT32;
typedef unsigned long long int UINT64;
typedef unsigned char UINT8;

UINT64 startTSC = 0;
UINT64 stopTSC = 0;
UINT64 cycleCnt = 0;

#define PMC_ASM(instructions,N,buf) \
  __asm__ __volatile__ ( instructions : "=A" (buf) : "c" (N) )

#define PMC_ASM_READ_TSC(buf) \
  __asm__ __volatile__ ( "rdtsc" : "=A" (buf) )

//#define PMC_ASM_READ_PMC(N,buf) PMC_ASM("rdpmc" "\n\t" "andl $255,%%edx",N,buf)
#define PMC_ASM_READ_PMC(N,buf) PMC_ASM("rdpmc",N,buf)

#define PMC_ASM_READ_CR(N,buf) \
  __asm__ __volatile__ ( "movl %%cr" #N ",%0" : "=r" (buf) )

UINT64 readTSC(void)
{
   UINT64 ts;

   __asm__ volatile(".byte 0x0f,0x31" : "=A" (ts));
   return ts;
}

UINT64 cyclesElapsed(UINT64 stopTS, UINT64 startTS)
{
   return (stopTS - startTS);
}

// PPM Edge Enhancement Code
UINT8 *header;
UINT8 *R;
UINT8 *G;
UINT8 *B;
UINT8 *convR;
UINT8 *convG;
UINT8 *convB;

#define FREE_ALL_MEM  free(header); free(R); free(G); free(B); free(convR); free(convG); free(convB);

#define K 4.0

FLOAT PSF[9] = {-K/8.0, -K/8.0, -K/8.0, -K/8.0, K+1.0, -K/8.0, -K/8.0, -K/8.0, -K/8.0};
UINT64 microsecs=0, clksPerMicro=0, millisecs=0;

void print_time_info (void)
{
   cycleCnt = cyclesElapsed(stopTSC, startTSC);
   microsecs = cycleCnt/clksPerMicro;
   millisecs = microsecs/1000;
   
   printf("Convolution time in cycles=%llu, rate=%llu, about %llu millisecs\n",
           cycleCnt, clksPerMicro, millisecs);
}

int main(int argc, char *argv[])
{
    int fdin, fdout, bytesRead=0, bytesLeft, i;
    FLOAT clkRate;
    int height = 0;
    int width = 0;
    int num_pixels = 0;

    int header_len = 0;

    // Estimate CPU clock rate
    startTSC = readTSC();
    usleep(1000000);
    stopTSC = readTSC();
    cycleCnt = cyclesElapsed(stopTSC, startTSC);

    printf("Cycle Count=%llu\n", cycleCnt);
    clkRate = ((FLOAT)cycleCnt)/1000000.0;
    clksPerMicro=(UINT64)clkRate;
    printf("Based on usleep accuracy, CPU clk rate = %llu clks/sec,",
          cycleCnt);
    printf(" %7.1f Mhz\n", clkRate);
    
    if(argc != 6)
    {
       printf("Usage: nored <file.ppm> <width> <height> <header_len> <outfile.ppm>\n");
       exit(-1);
    }
    else
    {
       width = atoi(argv[2]);
       height = atoi(argv[3]);
       header_len = atoi(argv[4]);
       num_pixels = width * height;

       printf("Using params: infile: %s, outfile: %s, height: %d, width: %d, header_len: %d\n", argv[1], argv[5], height, width, header_len);

       // Allocate memory for holding the pixels...
       header = malloc(header_len);
       R = malloc(num_pixels);
       G = malloc(num_pixels);
       B = malloc(num_pixels);
       convR = malloc(num_pixels);
       convG = malloc(num_pixels);
       convB = malloc(num_pixels);

       if(NULL == header ||
          NULL == R ||
          NULL == G ||
          NULL == B ||
          NULL == convR ||
          NULL == convB ||
          NULL == convG)
       {
          printf("Could not allocate the required memory!\n");
          exit(-1);
       }

       if((fdin = open(argv[1], O_RDONLY, 0644)) < 0)
       {
          printf("Error opening %s\n", argv[1]);
          exit(-1);
       }
       
       if((fdout = open(argv[5], (O_RDWR | O_CREAT), 0666)) < 0)
       {
          printf("Error opening %s\n", argv[1]);
          exit(-1);
       }
    }
    
    bytesLeft = header_len;

    do
    {
        //printf("bytesRead=%d, bytesLeft=%d\n", bytesRead, bytesLeft);
        bytesRead=read(fdin, (void *)header, bytesLeft);
        bytesLeft -= bytesRead;
    } while(bytesLeft > 0);

    header[header_len]='\0';

//    printf("header = %s\n", header);

/***************** Start of  core computation **************/
    // Start of convolution time stamp
    startTSC = readTSC();

    // Read RGB data
    for(i=0; i<num_pixels; i++)
    {
       read(fdin, (void *)&R[i], 1); 
       read(fdin, (void *)&G[i], 1);
       read(fdin, (void *)&B[i], 1);
       convR[i]=(11*R[i] + 16*G[i] + 5*B[i])/32;
       convG[i]=convR[i];
       convB[i]=convR[i];
    }

    // End of convolution time stamp
    stopTSC = readTSC();
/***************** End of core computation **************/

    print_time_info();

    write(fdout, (void *)header, header_len);

    // Write RGB data
    for(i=0; i<num_pixels; i++)
    {
        write(fdout, (void *)&convR[i], 1);
        write(fdout, (void *)&convG[i], 1);
        write(fdout, (void *)&convB[i], 1);
    }

    FREE_ALL_MEM;

    close(fdin);
    close(fdout);

    return 0;
}
