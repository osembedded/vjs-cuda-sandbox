#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdbool.h>
#include <string.h>

typedef double FLOAT;
//typedef float FLOAT;

typedef unsigned int UINT32;
typedef unsigned long long int UINT64;
typedef unsigned char UINT8;

UINT64 startTSC = 0;
UINT64 stopTSC = 0;
UINT64 cycleCnt = 0;

#define PRINT_PTR(x) printf("%s: %p\n", #x, x);

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
UINT8 *outfile;

#define K 4.0

FLOAT PSF[9] = {-K/8.0, -K/8.0, -K/8.0, -K/8.0, K+1.0, -K/8.0, -K/8.0, -K/8.0, -K/8.0};
UINT64 microsecs=0, clksPerMicro=0, millisecs=0;

/* User specified */
static char infile_pattern[128];
static char outfile_pattern[128];

void print_time_info (void)
{
   cycleCnt = cyclesElapsed(stopTSC, startTSC);
   microsecs = cycleCnt/clksPerMicro;
   millisecs = microsecs/1000;
   
   printf("Convolution time in cycles=%llu, rate=%llu, about %llu millisecs\n",
           cycleCnt, clksPerMicro, millisecs);
}

void read_ppm_header (int fd, int header_len)
{
   int bytesRead = 0;
   int bytesLeft = 0;
   
   if( -1 == fd )
   {
      printf("Invalid File pointer passed in! Exiting!\n");
      exit (-1);
   }

   bytesLeft = header_len;

   do
   {
      //printf("bytesRead=%d, bytesLeft=%d\n", bytesRead, bytesLeft);
      bytesRead=read(fd, (void *)header, bytesLeft);
      bytesLeft -= bytesRead;
   } while(bytesLeft > 0);
   
   header[header_len]='\0';
   
//    printf("header = %s\n", header);
}

bool interleave_components(UINT8 *ofile, int num_pix, UINT8 *RR, UINT8 *GG, UINT8 *BB)
{
   int retval = false;
   int ii = 0, jj = 0;

   if(NULL != ofile &&
      NULL != RR &&
      NULL != GG &&
      NULL != BB)
   {
      for(ii = 0; ii < num_pix; ii++)
      {
         ofile[jj++] = RR[ii];
         ofile[jj++] = GG[ii];
         ofile[jj++] = BB[ii];
      }

      retval = true;
   }

   return retval;
}

void write_output_to_file(int fdout, int num_pixels, int header_len)
{
   if( -1 == fdout )
   {
      printf("Invalid File pointer passed in! Exiting!\n");
      exit (-1);
   }

   write(fdout, (void *)header, header_len);
   
   if( true == interleave_components(outfile, num_pixels, convR, convG, convB))
   {
      write(fdout, (void *)outfile, num_pixels*3);
   }
}

bool open_files(int num, int* fdin, int* fdout)
{
   char in_file[256], out_file[256];

   if(NULL == fdin ||
      NULL == fdout)
   {
      printf("Null parameters passed in in open_files!\n");
      return false;
   }

   snprintf((char *)&in_file[0], 256, infile_pattern, num);
   snprintf((char *)&out_file[0], 256, outfile_pattern, num);
   
   if((*fdin = open((const char*)&in_file[0], O_RDONLY, 0644)) < 0)
   {
      printf("Error opening %s\n", in_file);
      return false;
   }
   
   if((*fdout = open((const char*)&out_file[0], (O_RDWR | O_CREAT), 0666)) < 0)
   {
      printf("Error opening %s\n", out_file);
      return false;
   }

   return true;
}

int main(int argc, char *argv[])
{
    int fdin, fdout, i;
    FLOAT clkRate;
    int height = 0;
    int width = 0;
    int num_pixels = 0;
    int header_len = 0;
    int seq_start_num = 0;
    int seq_count = 0;
    int jj = 0;

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
    
    if(argc != 8)
    {
       printf("Usage: blacknwhite <infile%%d.ppm> <width> <height> <header_len> <outfile%%d.ppm> <seq_start_num> <count>\n");
       exit(-1);
    }
    else
    {
       width = atoi(argv[2]);
       height = atoi(argv[3]);
       header_len = atoi(argv[4]);
       seq_start_num = atoi(argv[6]);
       seq_count = atoi(argv[7]);

       num_pixels = width * height;

       printf("Using params: infile pattern: %s, outfile pattern: %s, \nheight: %d, width: %d, header_len: %d, seq_start: %d, seq_count: %d\n", argv[1], argv[5], height, width, header_len, seq_start_num, seq_count);

       // Allocate memory for holding the pixels...
       header = malloc(header_len);
       R = malloc(num_pixels);
       G = malloc(num_pixels);
       B = malloc(num_pixels);
       convR = malloc(num_pixels);
       convG = malloc(num_pixels);
       convB = malloc(num_pixels);
       outfile = malloc(header_len + num_pixels*3);

       if(NULL == header ||
          NULL == R ||
          NULL == G ||
          NULL == B ||
          NULL == convR ||
          NULL == convB ||
          NULL == convG ||
          NULL == outfile)
       {
          printf("Could not allocate the required memory!\n");
          exit(-1);
       }

       strncpy(infile_pattern, argv[1], sizeof(infile_pattern));
       strncpy(outfile_pattern, argv[5], sizeof(outfile_pattern));
    }

    for(jj=seq_start_num; jj<(seq_start_num + seq_count); jj++)
    {
       if( false == open_files(jj, &fdin, &fdout))
       {
          printf("open files failed! bailing out!\n");
          break;
       }

       read_ppm_header(fdin, header_len);
       
/***************** Start of  core computation **************/
       startTSC = readTSC();
       
       // Read RGB data
       for(i=0; i<num_pixels; i++)
       {
          read(fdin, (void *)&R[i], 1); 
          read(fdin, (void *)&G[i], 1);
          read(fdin, (void *)&B[i], 1);
//          convR[i]=(11*R[i] + 16*G[i] + 5*B[i])/32;
          // Source: Wikipedia - http://en.wikipedia.org/wiki/Grayscale
          convR[i]=(0.30*R[i] + 0.59*G[i] + 0.11*B[i]);
          convG[i]=convR[i];
          convB[i]=convR[i];
       }
       
       stopTSC = readTSC();
       print_time_info();
/***************** End of core computation **************/

       startTSC = readTSC();

       write_output_to_file(fdout, num_pixels, header_len);

       stopTSC = readTSC();
       print_time_info();
       
       close(fdin);
       close(fdout);
    } // End of for loop.

    free(header); 
    free(R); 
    free(G); 
    free(B); 
    free(convR); 
    free(convG); 
    free(convB); 
    free(outfile);

    return 0;
}
