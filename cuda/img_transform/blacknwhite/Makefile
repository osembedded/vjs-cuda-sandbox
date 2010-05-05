################################################################################
#
# Build script for project
#
################################################################################
export OMIT_CUTIL_LIB=1
export dbg=1

BINDIR = ./

# Add source files here                                                                                                               
EXECUTABLE      := blacknwhite

# Cuda source files (compiled with cudacc)                                                                                            
CUFILES         := blacknwhite.cu

# C/C++ source files (compiled with gcc / c++)                                                                                        
CCFILES         := tsc.c

ROOTDIR := ~/NVIDIA_GPU_Computing_SDK/C/common

################################################################################
# Rules and targets                                                                                        
include ~/NVIDIA_GPU_Computing_SDK/C/common/common.mk

#CPPFLAGS = -O2 -g -Wall -Werror

cleanall:
	rm -rf blacknwhite obj release *~ core.*

help:
	@echo "***************************************"
	@echo "      Make Targets       "
	@echo "***************************************"
	@echo "make - "
	@echo "       Makes the blacknwhite executable."
	@echo "make cleanall - "
	@echo "       Cleans all objs and executable files."
	@echo "make clean - "
	@echo "       Beware - This deletes the tsc.c file."
	@echo "       If this happens, to get back the tsc.c"
	@echo "       file, run 'svn up tsc.c' ."
	@echo "***************************************"

#.PHONY: clean imgclean

#all: blacknwhite

#blacknwhite: blacknwhite.o

#clean:
#	rm -rf blacknwhite *.o *~ core.*

#imgclean: clean
#	rm -rf blacknwhite.ppm 
