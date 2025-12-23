NVCCFLAGS := -std=c++17 -gencode=arch=compute_100,code=\"sm_100,compute_100\"

# LDGSTS
CTAS_PER_SM ?= 1
NUM_STAGES ?= 1
THREADS_PER_BLOCK ?= 128
LDGSTS_DEFINE = -DCTAS_PER_SM=$(CTAS_PER_SM) -DNUM_STAGES=$(NUM_STAGES) -DTHREADS_PER_BLOCK=$(THREADS_PER_BLOCK)

# TMA 2D
SMEM_WIDTH ?=
SMEM_HEIGHT ?=
DEFINES = -DSMEM_WIDTH=$(SMEM_WIDTH) -DSMEM_HEIGHT=$(SMEM_HEIGHT)

# DSMEM Pair
CLUSTER_SIZE ?=
THREADS_PER_CTA ?=
LOAD_T ?=
DSMEM_PAIR_DEFS = -DCLUSTER_SIZE=$(CLUSTER_SIZE) -DTHREADS_PER_CTA=$(THREADS_PER_CTA) -DLOAD_T=$(LOAD_T)

all: cp_async tma

cp_async: cp_async.cu utils.h
	nvcc $(NVCCFLAGS) $(LDGSTS_DEFINE) -o $@ $<

cp_async.ptx: cp_async.cu utils.h
	nvcc $(NVCCFLAGS) $(LDGSTS_DEFINE) -ptx -o $@ $<

tma: tma.cu utils.h
	nvcc $(NVCCFLAGS) -o $@ $<

tma.ptx: tma.cu utils.h
	nvcc $(NVCCFLAGS) -ptx -o $@ $<

tma2d: tma2d.cu utils.h
	nvcc $(NVCCFLAGS) $(DEFINES) -lcuda -o $@ $<

tma2d.ptx: tma2d.cu utils.h
	nvcc $(NVCCFLAGS) $(DEFINES) -lcuda -ptx -o $@ $<

dsmem_pair: dsmem_pair.cu utils.h
	nvcc $(NVCCFLAGS) $(DSMEM_PAIR_DEFS) -o $@ $<

dsmem_pair.ptx: dsmem_pair.cu utils.h
	nvcc $(NVCCFLAGS) $(DSMEM_PAIR_DEFS) -ptx -o $@ $<

clean:
	rm -f *.ptx cp_async tma tma2d dsmem_pair
