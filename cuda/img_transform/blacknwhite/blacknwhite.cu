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
#define USE_CUDA

// PPM Edge Enhancement Code
UINT8 *header;
UINT8 *h_R;
UINT8 *h_G;
UINT8 *h_B;
UINT8 *d_R;
UINT8 *d_G;
UINT8 *d_B;
UINT8 *infile;
UINT8 *outfile;
UINT8 *frame_times;

#define PARAMS_GOOD                               \
   (NULL != header &&                             \
    NULL != h_R &&                                \
    NULL != h_G &&                                \
    NULL != h_B &&                                \
    NULL != d_R &&                                \
    NULL != d_B &&                                \
    NULL != d_G &&                                \
    NULL != infile &&                             \
    NULL != outfile &&                            \
    NULL != frame_times)

#ifdef USE_CUDA
#define FREE_MEM                                      \
   free(header);                                      \
   free(h_R);                                         \
   free(h_G);                                         \
   free(h_B);                                         \
   cudaFree(d_R);                                     \
   cudaFree(d_G);                                     \
   cudaFree(d_B);                                     \
   free(infile);                                      \
   free(outfile);                                     \
   free(frame_times);
#else
#define FREE_MEM                                      \
   free(header);                                      \
   free(h_R);                                         \
   free(h_G);                                         \
   free(h_B);                                         \
   free(d_R);                                         \
   free(d_G);                                         \
   free(d_B);                                         \
   free(infile);                                      \
   free(outfile);                                     \
   free(frame_times);
#endif // USE_CUDA


/* User specified */
static char infile_pattern[128];
static char outfile_pattern[128];

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

void write_output_to_file(int fdout, int num_pixels, int header_len,
                          UINT8 *RR, UINT8 *GG, UINT8 *BB)
{
   if( -1 == fdout )
   {
      printf("Invalid File pointer passed in! Exiting!\n");
      exit (-1);
   }

   if( NULL == RR ||
       NULL == GG ||
       NULL == BB )
   {
      printf("NULL parameters passed in! exiting!\n");
      exit (-1);
   }

   write(fdout, (void *)header, header_len);
   
   if( true == interleave_components(outfile, num_pixels, 
                                     RR, GG, BB))
   {
      write(fdout, (void *)outfile, num_pixels*3);
   }
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
         h_R[ii] = ifile[jj++];
         h_G[ii] = ifile[jj++];
         h_B[ii] = ifile[jj++];         
      }

      retval = true;
   }

   return retval;
}

void read_input_from_file(int fdin, int num_pixels, int header_len,
                          UINT8 *RR, UINT8 *GG, UINT8 *BB)
{
   if( -1 == fdin )
   {
      printf("Invalid File pointer passed in! Exiting!\n");
      exit (-1);
   }

   if( NULL == RR ||
       NULL == GG ||
       NULL == BB )
   {
      printf("NULL parameters passed in! exiting!\n");
      exit (-1);
   }

   save_ppm_header(fdin, header_len);

   read(fdin, (void *)infile, num_pixels*3);

   separate_components(infile, num_pixels, 
                       RR, GG, BB);
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

void transform_pixels (UINT8 *h_RR, UINT8 *h_GG, UINT8 *h_BB,
                       UINT8 *d_RR, UINT8 *d_GG, UINT8 *d_BB,
                       int NN)

{
   int block_size = 512;
   dim3 dimBlock(block_size);
   dim3 dimGrid(NN/block_size);

//   printf("block Size = %d, NN = %d\n", block_size, NN);
//   printf("Grid Size = %d\n",NN/block_size);

   cudaMemcpy(d_RR, h_RR, NN, cudaMemcpyHostToDevice);
   cudaMemcpy(d_GG, h_GG, NN, cudaMemcpyHostToDevice);
   cudaMemcpy(d_BB, h_BB, NN, cudaMemcpyHostToDevice);

   cudaKernel<<<dimGrid, dimBlock>>>(d_RR, d_GG, d_BB, NN);
   cudaThreadSynchronize();

   cudaMemcpy(h_RR, d_RR, NN, cudaMemcpyDeviceToHost);
   cudaMemcpy(h_GG, d_GG, NN, cudaMemcpyDeviceToHost);
   cudaMemcpy(h_BB, d_BB, NN, cudaMemcpyDeviceToHost);
}

#else

void convert_to_grayscale (UINT8 *Rin, UINT8 *Gin, UINT8 *Bin,
                           UINT8 *Rout, UINT8 *Gout, UINT8 *Bout,
                           int NN)
{
   int ii = 0;

   // Read RGB data
   for(ii = 0; ii < NN; ii++)
   {
      // Source: Wikipedia - http://en.wikipedia.org/wiki/Grayscale
      Rout[ii]=( 0.30 * Rin[ii] ) + ( 0.59 * Gin[ii] ) + ( 0.11 * Bin[ii] );
      Gout[ii]=Rout[ii];
      Bout[ii]=Rout[ii];
   }
}

void transform_pixels (UINT8 *Rin, UINT8 *Gin, UINT8 *Bin,
                       UINT8 *Rout, UINT8 *Gout, UINT8 *Bout,
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
      totalTime += frame_times[ii];
   }

   printf("Total time taken to process %d frames: %llu mSecs\n", num_frames, totalTime);
   printf("Average time per frame: %llu mSecs\n", totalTime/num_frames);
}


#define NUM_ARGS (8)
int main(int argc, char *argv[])
{
   int fdin, fdout;
   int height = 0;
   int width = 0;
   int num_pixels = 0;
   int header_len = 0;
   int seq_start_num = 0;
   int seq_count = 0;
   int jj = 0;

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
      h_R = (UINT8 *) malloc(num_pixels);
      h_G = (UINT8 *) malloc(num_pixels);
      h_B = (UINT8 *) malloc(num_pixels);

#ifdef USE_CUDA
      cudaMalloc((void **) &d_R, num_pixels);
      cudaMalloc((void **) &d_G, num_pixels);
      cudaMalloc((void **) &d_B, num_pixels);
#else
      // Note: Even though these are named 'd_' for device memory,
      // In the case of NON CUDA code, we allocate them from the host.
      d_R = (UINT8 *) malloc(num_pixels);
      d_G = (UINT8 *) malloc(num_pixels);
      d_B = (UINT8 *) malloc(num_pixels);
#endif

      outfile = (UINT8 *) malloc(header_len + num_pixels*3);
      infile = (UINT8 *) malloc(header_len + num_pixels*3);

      // Allocate memory for computation.
      frame_times = (UINT8 *) malloc(seq_count);

      if(true != PARAMS_GOOD)
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

      read_input_from_file(fdin, num_pixels, header_len,
                           h_R, h_G, h_B);

/***************** Start of  core computation **************/
      save_start_time();
      transform_pixels(h_R, h_G, h_B,
                       d_R, d_G, d_B,
                       num_pixels);

      save_stop_time();
      frame_times[jj] = calc_time_diff();
/***************** End of core computation **************/

#ifdef USE_CUDA
      write_output_to_file(fdout, num_pixels, header_len,
                           h_R, h_G, h_B);
#else
      write_output_to_file(fdout, num_pixels, header_len,
                           d_R, d_G, d_B);
#endif

      close(fdin);
      close(fdout);

   } // Loop through sequence of images

   print_time_stats(jj);

   FREE_MEM;

   return 0;
}
