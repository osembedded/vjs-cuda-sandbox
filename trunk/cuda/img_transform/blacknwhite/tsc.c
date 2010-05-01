#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "types.h"

static UINT64 startTSC;
static UINT64 stopTSC;
static UINT64 cycleCnt;
static UINT64 microsecs;
static UINT64 clksPerMicro;
static UINT64 millisecs;

#define PMC_ASM(instructions,N,buf)                             \
   __asm__ __volatile__ ( instructions : "=A" (buf) : "c" (N) )

#define PMC_ASM_READ_TSC(buf)                           \
   __asm__ __volatile__ ( "rdtsc" : "=A" (buf) )

//#define PMC_ASM_READ_PMC(N,buf) PMC_ASM("rdpmc" "\n\t" "andl $255,%%edx",N,buf)
#define PMC_ASM_READ_PMC(N,buf) PMC_ASM("rdpmc",N,buf)

#define PMC_ASM_READ_CR(N,buf)                                  \
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

UINT64 calc_time_diff(void)
{
   cycleCnt = cyclesElapsed(stopTSC, startTSC);
   microsecs = cycleCnt/clksPerMicro;
   millisecs = microsecs/1000;
   return millisecs;
}

void save_start_time(void)
{
   startTSC = readTSC();
}

void save_stop_time(void)
{
   stopTSC = readTSC();
}

FLOAT estimate_clk_rate (void)
{
   FLOAT clkRate = 0.0;

   save_start_time();
   usleep(1000000);
   save_stop_time();

   cycleCnt = cyclesElapsed(stopTSC, startTSC);

   printf("Cycle Count=%llu\n", cycleCnt);
   clkRate = ((FLOAT)cycleCnt)/1000000.0;
   clksPerMicro=(UINT64)clkRate;
   printf("Based on usleep accuracy, CPU clk rate = %llu clks/sec,",
          cycleCnt);
   printf(" %7.1f Mhz\n", clkRate);

   return clkRate;
}
