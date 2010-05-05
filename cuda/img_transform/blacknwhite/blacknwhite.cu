#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdbool.h>
#include <string.h>

#include "types.h"
#include "tsc.h"

// Turn this switch on if you want to 
// use cuda based acceleration...
//#define USE_CUDA

#define NUM_SIMUL_FRAMES (4)

// PPM Edge Enhancement Code
UINT8 *header;
UINT8 *h_R[NUM_SIMUL_FRAMES];
UINT8 *h_G[NUM_SIMUL_FRAMES];
UINT8 *h_B[NUM_SIMUL_FRAMES];
UINT8 *d_R[NUM_SIMUL_FRAMES];
UINT8 *d_G[NUM_SIMUL_FRAMES];
UINT8 *d_B[NUM_SIMUL_FRAMES];
UINT8 *infile;
UINT8 *outfile;
int *frame_times;

#define PARAMS_GOOD                             \
   (NULL != header &&                           \
    NULL != h_R &&                              \
    NULL != h_G &&                              \
    NULL != h_B &&                              \
    NULL != d_R &&                              \
    NULL != d_B &&                              \
    NULL != d_G &&                              \
    NULL != infile &&                           \
    NULL != outfile &&                          \
    NULL != frame_times)

#ifdef USE_CUDA
#define FREE_MEM                                \
   free(header);                                \
   for(int xx=0; xx< NUM_SIMUL_FRAMES; xx++)    \
   {                                            \
      free(h_R[xx]);                            \
      free(h_G[xx]);                            \
      free(h_B[xx]);                            \
      cudaFree(d_R[xx]);                        \
      cudaFree(d_G[xx]);                        \
      cudaFree(d_B[xx]);                        \
   }                                            \
   free(infile);                                \
   free(outfile);                               \
   free(frame_times);
#else
#define FREE_MEM                                \
   free(header);                                \
   for(int xx=0; xx< NUM_SIMUL_FRAMES; xx++)    \
   {                                            \
      free(h_R[xx]);                            \
      free(h_G[xx]);                            \
      free(h_B[xx]);                            \
      free(d_R[xx]);                            \
      free(d_G[xx]);                            \
      free(d_B[xx]);                            \
   }                                            \
   free(infile);                                \
   free(outfile);                               \
   free(frame_times);
#endif // USE_CUDA


/* User specified */
static char infile_pattern[128];
static char outfile_pattern[128];

/* Function prototypes */
bool open_output_file(int num, int* fdout);
bool open_input_file(int num, int* fdin);

void save_ppm_header (int fd, int header_len)
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

bool interleave_components(UINT8 *ofile, int num_pix, 
                           UINT8 *RR, UINT8 *GG, UINT8 *BB)
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
         // This is where it seg faults if we mess up the memory access...
         ofile[jj++] = RR[ii];
         ofile[jj++] = GG[ii];
         ofile[jj++] = BB[ii];
      }

      retval = true;
   }

   return retval;
}

bool write_output_to_file(int file_num, int num_pixels, int header_len,
                          UINT8 *RR, UINT8 *GG, UINT8 *BB)
{
   bool retval = false;
   int fdout = -1;

   if( false == open_output_file(file_num, &fdout))
   {
      printf("Could not open output file!!\n");
   }
   else if( NULL == RR ||
            NULL == GG ||
            NULL == BB )
   {
      printf("NULL parameters passed in!\n");
   }
   else
   {
      write(fdout, (void *)header, header_len);
      
      if( true == interleave_components(outfile, num_pixels, 
                                        RR, GG, BB))
      {
         write(fdout, (void *)outfile, num_pixels*3);
      }

      retval = true;
   }

   if(fdout)
      close(fdout);

   return retval;
}
  
bool separate_components (UINT8 *ifile, int num_pix, 
                          UINT8 *RR, UINT8 *GG, UINT8 *BB)
{
   int retval = false;
   int ii = 0, jj = 0;

   if(NULL != ifile &&
      NULL != RR &&
      NULL != GG &&
      NULL != BB)
   {
      for(ii = 0; ii < num_pix; ii++)
      {
         RR[ii] = ifile[jj++];
         GG[ii] = ifile[jj++];
         BB[ii] = ifile[jj++];         
      }

      retval = true;
   }

   return retval;
}

bool read_input_from_file(int file_num, int num_pixels, int header_len,
                          UINT8 *RR, UINT8 *GG, UINT8 *BB)
{
   bool retval = false;
   int fdin = -1;

   if( false == open_input_file(file_num, &fdin))
   {
      printf("open input file failed! bailing out!\n");
   }
   else if( NULL == RR ||
            NULL == GG ||
            NULL == BB )
   {
      printf("NULL parameters passed in! exiting!\n");
   }
   else
   {
      save_ppm_header(fdin, header_len);
      
      read(fdin, (void *)infile, num_pixels*3);
      
      separate_components(infile, num_pixels, 
                          RR, GG, BB);

      retval = true;
   }

   if(-1 != fdin)
      close(fdin);

   return retval;
}

bool open_input_file(int num, int* fdin)
{
   char in_file[256];

   if(NULL == fdin)
   {
      printf("Null parameter passed in open_files!\n");
      return false;
   }

   snprintf((char *)&in_file[0], 256, infile_pattern, num);

//   printf("Processing File %s\n", in_file);
   
   if((*fdin = open((const char*)&in_file[0], O_RDONLY, 0644)) < 0)
   {
      printf("Error opening %s\n", in_file);
      return false;
   }

   return true;
}

bool open_output_file(int num, int* fdout)
{
   char out_file[256];

   if(NULL == fdout)
   {
      printf("Null parameter passed in open_files!\n");
      return false;
   }

   snprintf((char *)&out_file[0], 256, outfile_pattern, num);
   
   if((*fdout = open((const char*)&out_file[0], (O_RDWR | O_CREAT), 0666)) < 0)
   {
      printf("Error opening %s\n", out_file);
      return false;
   }

   return true;
}

#ifdef USE_CUDA

/* Our main cuda kernel */
__global__ void cudaKernel (UINT8 *RR, UINT8 *GG, UINT8 *BB,
                            int NN)
{
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   
   if(idx < NN)
   {
      RR[idx] = (0.30 * RR[idx]) + (0.59 * GG[idx]) + (0.11 * BB[idx]);
      GG[idx] = RR[idx];
      BB[idx] = RR[idx];
   }
}

void transform_pixels (UINT8 **h_RR, UINT8 **h_GG, UINT8 **h_BB,
                       UINT8 **d_RR, UINT8 **d_GG, UINT8 **d_BB,
                       int NN)

{
   int block_size = 512;
   dim3 dimBlock(block_size);
   dim3 dimGrid(NN/block_size);
   int ii = 0;

//   printf("block Size = %d, NN = %d\n", block_size, NN);
//   printf("Grid Size = %d\n",NN/block_size);
   
   for(ii=0; ii< NUM_SIMUL_FRAMES; ii++)
   {
      cudaMemcpy(d_RR[ii], h_RR[ii], NN, cudaMemcpyHostToDevice);
      cudaMemcpy(d_GG[ii], h_GG[ii], NN, cudaMemcpyHostToDevice);
      cudaMemcpy(d_BB[ii], h_BB[ii], NN, cudaMemcpyHostToDevice);

      cudaKernel<<<dimGrid, dimBlock>>>(d_RR[ii], d_GG[ii], d_BB[ii], NN);
      cudaThreadSynchronize();
      
      cudaMemcpy(h_RR[ii], d_RR[ii], NN, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_GG[ii], d_GG[ii], NN, cudaMemcpyDeviceToHost);
      cudaMemcpy(h_BB[ii], d_BB[ii], NN, cudaMemcpyDeviceToHost);
   }
}

#else

void convert_to_grayscale (UINT8 **Rin, UINT8 **Gin, UINT8 **Bin,
                           UINT8 **Rout, UINT8 **Gout, UINT8 **Bout,
                           int NN)
{
   int ii = 0;
   int jj = 0;

   for(jj=0; jj < NUM_SIMUL_FRAMES; jj++)
   {
      // Read RGB data
      for(ii = 0; ii < NN; ii++)
      {
         // Source: Wikipedia - http://en.wikipedia.org/wiki/Grayscale
         Rout[jj][ii]=( 0.30 * Rin[jj][ii] ) + ( 0.59 * Gin[jj][ii] ) + ( 0.11 * Bin[jj][ii] );
         Gout[jj][ii]=Rout[jj][ii];
         Bout[jj][ii]=Rout[jj][ii];
      }
   }
}

void transform_pixels (UINT8 **Rin, UINT8 **Gin, UINT8 **Bin,
                       UINT8 **Rout, UINT8 **Gout, UINT8 **Bout,
                       int NN)
{
   convert_to_grayscale(Rin, Gin, Bin,
                        Rout, Gout, Bout,
                        NN);
}

#endif // USE_CUDA

void print_time_stats(int num_frames)
{
   int ii = 0;
   UINT64 totalTime = 0;

   if(0 == num_frames)
   {
      printf("No frames processed! Exiting!\n");
      exit (-1);
   }

   for(ii = 0; ii < num_frames; ii++)
   {
      printf("frame_times[%d]: %d\n", ii, frame_times[ii]);
      totalTime += frame_times[ii];
   }

   printf("Total time taken to process %d frames: %llu mSecs\n", num_frames, totalTime);
   printf("Average time per frame: %llu mSecs\n", totalTime/num_frames);
}


#define NUM_ARGS (8)
int main(int argc, char *argv[])
{
   int height = 0;
   int width = 0;
   int num_pixels = 0;
   int header_len = 0;
   int seq_start_num = 0;
   int seq_count = 0;
   int jj = 0;
   int ii = 0;

   // Estimate CPU clock rate
   estimate_clk_rate();
    
   if(argc != NUM_ARGS)
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

      printf("Using params: infile pattern: %s, outfile pattern: %s, \nheight: %d, width: %d, header_len: %d, seq_start: %d, seq_count: %d\n", 
             argv[1], argv[5], height, width, header_len, seq_start_num, seq_count);

      // Allocate memory for holding the pixels...
      header = (UINT8 *) malloc(header_len);

      for(ii = 0; ii < NUM_SIMUL_FRAMES; ii++)
      {
         h_R[ii] = (UINT8 *) malloc(num_pixels);
         h_G[ii] = (UINT8 *) malloc(num_pixels);
         h_B[ii] = (UINT8 *) malloc(num_pixels);

#ifdef USE_CUDA
         cudaMalloc((void **) &d_R[ii], num_pixels);
         cudaMalloc((void **) &d_G[ii], num_pixels);
         cudaMalloc((void **) &d_B[ii], num_pixels);
#else
         // Note: Even though these are named 'd_' for device memory,
         // In the case of NON CUDA code, we allocate them from the host.
         d_R[ii] = (UINT8 *) malloc(num_pixels);
         d_G[ii] = (UINT8 *) malloc(num_pixels);
         d_B[ii] = (UINT8 *) malloc(num_pixels);
#endif // USE_CUDA

         outfile = (UINT8 *) malloc(header_len + num_pixels*3);
         infile = (UINT8 *) malloc(header_len + num_pixels*3);
         
         // Allocate memory for computation.
         frame_times = (int *) calloc(seq_count, sizeof(int));
         
         if(true != PARAMS_GOOD)
         {
            printf("Could not allocate the required memory!\n");
            exit(-1);
         }
         
         strncpy(infile_pattern, argv[1], sizeof(infile_pattern));
         strncpy(outfile_pattern, argv[5], sizeof(outfile_pattern));
      }
      
      for(jj=seq_start_num; jj<(seq_start_num + seq_count); jj=jj+NUM_SIMUL_FRAMES)
      {
         for(ii=0; ii < NUM_SIMUL_FRAMES; ii++)
         {
//            printf("jj+ii = %d\n", jj+ii);
            if(false == read_input_from_file(jj+ii, num_pixels, header_len,
                                             h_R[ii], h_G[ii], h_B[ii]))
            {
               exit (-1);
            }
         }
         
/***************** Start of  core computation **************/
         save_start_time();
         transform_pixels(h_R, h_G, h_B,
                          d_R, d_G, d_B,
                          num_pixels);
         
         save_stop_time();
         frame_times[jj] = calc_time_diff();
/***************** End of core computation **************/
         
         for(ii=0; ii < NUM_SIMUL_FRAMES; ii++)
         {
            bool tmpval;
#ifdef USE_CUDA
            tmpval = write_output_to_file(jj+ii, num_pixels, header_len,
                                          h_R[ii], h_G[ii], h_B[ii]);
#else
            tmpval = write_output_to_file(jj+ii, num_pixels, header_len,
                                          d_R[ii], d_G[ii], d_B[ii]);
#endif
         
            if(false == tmpval)
            {
               exit(-1);
            }
         }
      
      } // Loop through sequence of images

      //Note: If you make the above more intelligent such that 
      //if you fail some file open or close, you don't exit 
      //but instead break the outer for loop, we can still 
      //compute the time it took based on the count jj. For 
      //now though, I am keeping this simple...
      print_time_stats(seq_count);
      
      FREE_MEM;
   }
   
   return 0;
}
