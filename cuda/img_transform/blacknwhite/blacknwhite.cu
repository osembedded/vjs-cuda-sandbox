#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdbool.h>
#include <string.h>

#include "types.h"
#include "tsc.h"

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

/* User specified */
static char infile_pattern[128];
static char outfile_pattern[128];

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
    int height = 0;
    int width = 0;
    int num_pixels = 0;
    int header_len = 0;
    int seq_start_num = 0;
    int seq_count = 0;
    int jj = 0;

    // Estimate CPU clock rate
    estimate_clk_rate();
    
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
       header = (UINT8 *) malloc(header_len);
       R = (UINT8 *) malloc(num_pixels);
       G = (UINT8 *) malloc(num_pixels);
       B = (UINT8 *) malloc(num_pixels);
       convR = (UINT8 *) malloc(num_pixels);
       convG = (UINT8 *) malloc(num_pixels);
       convB = (UINT8 *) malloc(num_pixels);
       outfile = (UINT8 *) malloc(header_len + num_pixels*3);

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
       save_start_time();
       
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
       
       save_stop_time();
       print_time_info();
/***************** End of core computation **************/
       save_start_time();
       write_output_to_file(fdout, num_pixels, header_len);
       save_stop_time();
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
