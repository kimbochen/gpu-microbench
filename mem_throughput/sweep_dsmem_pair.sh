#!/bin/bash

NCU_BIN=$(which ncu)

CLUSTER_SIZES=(2 4 8)
THREADS_PER_CTA=(128 256 512 1024)
LOAD_TYPES=("float" "float2" "float4")

OUTPUT_FILE="dsmem_pair_sweep_results.csv"

# echo "Cluster Size,Threads per CTA,Load per Thread (B),Instruction Throughput (instr/s)" > "$OUTPUT_FILE"

# Map LOAD_T to bytes
get_load_bytes() {
    case "$1" in
        "float")  echo 4 ;;
        "float2") echo 8 ;;
        "float4") echo 16 ;;
    esac
}

for cluster_size in "${CLUSTER_SIZES[@]}"; do
    for threads in "${THREADS_PER_CTA[@]}"; do
        for load_t in "${LOAD_TYPES[@]}"; do
            load_bytes=$(get_load_bytes "$load_t")
            echo "Running: CLUSTER_SIZE=$cluster_size THREADS_PER_CTA=$threads LOAD_T=$load_t ($load_bytes B)"

            make clean
            make dsmem_pair CLUSTER_SIZE="$cluster_size" THREADS_PER_CTA="$threads" LOAD_T="$load_t"

            INSTR_PER_SEC=$(
                sudo "$NCU_BIN" --clock-control none --csv --metrics sm__sass_inst_executed_op_dshared_ld.sum.per_second ./dsmem_pair 2>/dev/null \
                | tail -1 | awk -F'","' '{print $NF}' | tr -d '"'
            )

            echo "DSMEM Instruction Throughput: ${INSTR_PER_SEC} instr/s"
            echo "${cluster_size},${threads},${load_bytes},${INSTR_PER_SEC}" >> "$OUTPUT_FILE"
        done
    done
done
