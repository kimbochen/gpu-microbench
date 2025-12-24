#!/bin/bash

NCU_BIN=$(which ncu)
OUTPUT_FILE="dsmem_ring_sweep_results.csv"

CLUSTER_SIZES=(2 4 8)
THREADS_PER_CTA=(128 256 512 1024)
LOAD_TYPES=("float" "float2" "float4")

echo "Cluster Size,Threads per CTA,Load per Thread (B),Memory Throughput (GB/s)" > "$OUTPUT_FILE"

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
            make dsmem_ring CLUSTER_SIZE="$cluster_size" THREADS_PER_CTA="$threads" LOAD_T="$load_t"

            BYTES_PER_SEC=$(
                sudo "$NCU_BIN" --clock-control none --csv --metrics sm__sass_data_bytes_mem_shared_op_ld.avg.per_second ./dsmem_ring 2>/dev/null \
                | tail -1 | awk -F'","' '{print $NF}' | tr -d '"'
            )
            GB_PER_SEC=$(echo "$BYTES_PER_SEC" | awk '{printf "%.4f", $1 / 1e9}')

            echo "DSMEM Ring Memory Throughput: ${GB_PER_SEC} GB/s"
            echo "${cluster_size},${threads},${load_bytes},${GB_PER_SEC}" >> "$OUTPUT_FILE"
        done
    done
done
