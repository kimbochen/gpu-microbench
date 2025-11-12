NVCCFLAGS   := -std=c++11 -gencode=arch=compute_100,code=\"sm_100,compute_100\" 

all: ldg cp_async tma

ldg: ldg.cu
	nvcc $(NVCCFLAGS) -o $@ $<

ldg.ptx: ldg.cu
	nvcc $(NVCCFLAGS) -ptx -o $@ $<

cp_async: cp_async.cu
	nvcc $(NVCCFLAGS) -o $@ $<

cp_async.ptx: cp_async.cu
	nvcc $(NVCCFLAGS) -ptx -o $@ $<

tma: tma.cu
	nvcc $(NVCCFLAGS) -o $@ $<

tma.ptx: tma.cu
	nvcc $(NVCCFLAGS) -ptx -o $@ $<

clean:
	rm -f *.ptx ldg cp_async tma
