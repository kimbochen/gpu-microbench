GENCODE_SM50 := -gencode=arch=compute_50,code=\"sm_50,compute_50\"
GENCODE_SM61 := -gencode=arch=compute_61,code=\"sm_61,compute_61\"
GENCODE_SM70 := -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM80 := -gencode=arch=compute_80,code=\"sm_80,compute_80\"
GENCODE_SM89 := -gencode=arch=compute_89,code=\"sm_89,compute_89\"
GENCODE_SM90 := -gencode=arch=compute_90,code=\"sm_90,compute_90\"
GENCODE_SM100 := -gencode=arch=compute_100,code=\"sm_100,compute_100\"

# Usage: make ARRAY_SIZE=$((D*131072)) && ./mem_lat, D: Chain data volume (MB)

CUOPTS = $(GENCODE_SM100)

CC := nvcc

INCLUDE := 
LIB := 
ARRAY_SIZE ?= 1048576

SRC = mem_lat.cu

EXE = mem_lat

NVCC_FLAGS = #-Xptxas -dlcm=cv -Xptxas -dscm=wt

release:
	$(CC) $(NVCC_FLAGS) $(CUOPTS) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) -DARRAY_SIZE=$(ARRAY_SIZE) -lcudart

run:
	./$(EXE)

profile:
	nvprof ./$(EXE)

events:
	nvprof  --events elapsed_cycles_sm ./$(EXE)
