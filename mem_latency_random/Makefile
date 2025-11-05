NVCCFLAGS   :=  -std=c++11 -arch=sm_100
CCFLAGS     :=
LDFLAGS     := -lcublas -lnvidia-ml


ld_global: ld_global.cu gpu-clock.cuh
	nvcc $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

ld_global.ptx: ld_global.cu gpu-clock.cuh
	nvcc $(NVCCFLAGS) $(INCLUDES) -ptx -o $@ $< $(LDFLAGS)

tma: tma.cu gpu-clock.cuh
	nvcc $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

tma.ptx: tma.cu gpu-clock.cuh
	nvcc $(NVCCFLAGS) $(INCLUDES) -ptx -o $@ $< $(LDFLAGS)

cp_async: cp_async.cu gpu-clock.cuh
	nvcc $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

cp_async.ptx: cp_async.cu gpu-clock.cuh
	nvcc $(NVCCFLAGS) $(INCLUDES) -ptx -o $@ $< $(LDFLAGS)

clean:
	rm -f *.ptx ld_global cp_async tma
