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

# DSMEM
CLUSTER_SIZE ?=
THREADS_PER_CTA ?=
LOAD_T ?=
DSMEM_DEFS = -DCLUSTER_SIZE=$(CLUSTER_SIZE) -DTHREADS_PER_CTA=$(THREADS_PER_CTA) -DLOAD_T=$(LOAD_T)

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
	nvcc $(NVCCFLAGS) $(DSMEM_DEFS) -o $@ $<

dsmem_pair.ptx: dsmem_pair.cu utils.h
	nvcc $(NVCCFLAGS) $(DSMEM_DEFS) -ptx -o $@ $<

dsmem_ring: dsmem_ring.cu utils.h
	nvcc $(NVCCFLAGS) $(DSMEM_DEFS) -o $@ $<

dsmem_ring.ptx: dsmem_ring.cu utils.h
	nvcc $(NVCCFLAGS) $(DSMEM_DEFS) -ptx -o $@ $<

dsmem_bcast: dsmem_bcast.cu utils.h
	nvcc $(NVCCFLAGS) $(DSMEM_DEFS) -o $@ $<

dsmem_bcast.ptx: dsmem_bcast.cu utils.h
	nvcc $(NVCCFLAGS) $(DSMEM_DEFS) -ptx -o $@ $<

dsmem_local: dsmem_local.cu utils.h
	nvcc $(NVCCFLAGS) $(DSMEM_DEFS) -o $@ $<

dsmem_local.ptx: dsmem_local.cu utils.h
	nvcc $(NVCCFLAGS) $(DSMEM_DEFS) -ptx -o $@ $<

clean:
	rm -f *.ptx cp_async tma tma2d dsmem_pair dsmem_ring dsmem_bcast dsmem_local
