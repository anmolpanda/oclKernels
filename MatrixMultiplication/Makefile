CC = gcc
CCFLAGS = -g  -Wall -O3  -DDEBUG
CL_FLAGS = -lOpenCL 
OPT = -lm -lrt

UNAME = $(shell uname -s)
#$(info $(UNAME))
OSARCH = $(shell uname -m | grep "_64")

ifeq ($(UNAME),Linux)
  CUDADIR = $(shell ls /usr/local/ | grep cuda )
  AMDAPPDIR = $(shell ls /opt/ | grep AMDAPP )
  ifneq ($(findstring cuda,$(CUDADIR)),)
    CL_INCLUDE = -I/usr/local/cuda/include
    ifneq ($(findstring _64, $(OSARCH)),)
        CL_LIBPATH = -L/usr/local/cuda/lib64 
    else
        CL_LIBPATH = -L/usr/local/cuda/lib 
    endif
  else 
    ifneq ($(findstring AMDAPP, $(AMDAPPDIR)),)
      CL_INCLUDE = -I/opt/AMDAPP/include
      ifneq ($(findstring _64, $(OSARCH)),)
        CL_LIBPATH = -L/opt/AMDAPP/lib/x86_64 
      else
        CL_LIBPATH = -L/opt/AMDAPP/lib/x86 
      endif
    else
      $(error CUDA / AMDAPP toolkit are not installed on Linux!) 
    endif
  endif
else
  # windows, osx are not supported
  $(error Windows and OSX are currently not supported, expect Linux!) 
endif

SRCS =  $(wildcard *.c)
PROG =  $(SRCS:.c=)

all: c_utils.o ocl_utils.o mm	

#--------------
# build objects
#--------------
c_utils.o: ../Utils/c_utils.h ../Utils/c_utils.c 
	$(CC) $(CCFLAGS) -c ../Utils/c_utils.c -o ../Utils/c_utils.o  $(OPT)

ocl_utils.o: ../Utils/ocl_utils.h ../Utils/ocl_utils.c 
	$(CC) $(CCFLAGS) $(CL_INCLUDE) $(CL_LIBPATH) -c ../Utils/ocl_utils.c -o ../Utils/ocl_utils.o $(CL_FLAGS) $(OPT)



#--------------
# build programs 
#--------------
mm:  mm.c
	$(CC) $(CCFLAGS) $(CL_INCLUDE) $(CL_LIBPATH) mm.c -o mm ../Utils/c_utils.o  ../Utils/ocl_utils.o $(CL_FLAGS) $(OPT)


.PHONY: clean
clean:
	rm -rf $(PROG) ../Utils/*.o 
