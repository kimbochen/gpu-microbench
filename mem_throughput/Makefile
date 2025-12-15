NVCCFLAGS := -std=c++17 -gencode=arch=compute_100,code=\"sm_100,compute_100\"

# TMA 2D
SMEM_WIDTH ?=
SMEM_HEIGHT ?=
DEFINES = -DSMEM_WIDTH=$(SMEM_WIDTH) -DSMEM_HEIGHT=$(SMEM_HEIGHT)

all: cp_async tma

cp_async: cp_async.cu utils.h
	nvcc $(NVCCFLAGS) -o $@ $<

cp_async.ptx: cp_async.cu utils.h
	nvcc $(NVCCFLAGS) -ptx -o $@ $<

tma: tma.cu utils.h
	nvcc $(NVCCFLAGS) -o $@ $<

tma.ptx: tma.cu utils.h
	nvcc $(NVCCFLAGS) -ptx -o $@ $<

tma2d: tma2d.cu utils.h
	nvcc $(NVCCFLAGS) $(DEFINES) -lcuda -o $@ $<

tma2d.ptx: tma2d.cu utils.h
	nvcc $(NVCCFLAGS) $(DEFINES) -lcuda -ptx -o $@ $<

clean:
	rm -f *.ptx cp_async tma tma2d
