#ifndef TSC_H
#define TSC_H

UINT64 readTSC(void);
UINT64 cyclesElapsed(UINT64 stopTS, UINT64 startTS);
void save_start_time(void);
void save_stop_time(void);
FLOAT estimate_clk_rate (void);
UINT64 calc_time_diff(void);
#endif //TSC_H
