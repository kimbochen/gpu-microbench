#!/bin/bash

NCU_BIN=$(which ncu)
OUTPUT_FILE="tma2d_sweep_results.csv"
COMBINATIONS=(
    "32 4"
    "64 2"
    "128 1"
    "32 16"
    "64 8"
    "128 4"
    "256 2"
    "32 32"
    "64 16"
    "128 8"
    "256 4"
    "32 64"
    "64 32"
    "128 16"
    "256 8"
    "32 128"
    "64 64"
    "128 32"
    "256 16"
    "32 192"
    "48 128"
    "64 96"
    "128 48"
    "192 32"
    "256 24"
)

echo "SMEM_WIDTH,SMEM_HEIGHT,dram_read_TB_per_s" > "$OUTPUT_FILE"

for combo in "${COMBINATIONS[@]}"; do
    read -r WIDTH HEIGHT <<< "$combo"
    echo "Running: SMEM_WIDTH=$WIDTH SMEM_HEIGHT=$HEIGHT"

    make clean
    make tma2d SMEM_WIDTH="$WIDTH" SMEM_HEIGHT="$HEIGHT"
    BYTES_PER_SEC=$(
        sudo "$NCU_BIN" --clock-control none --csv --metrics dram__bytes_read.sum.per_second ./tma2d 2>/dev/null \
        | tail -1 | awk -F'","' '{print $NF}' | tr -d '"'
    )
    TB_PER_SEC=$(echo "$BYTES_PER_SEC" | awk '{printf "%.4f", $1 / 1e12}')

    echo "  dram_read: ${TB_PER_SEC} TB/s"
    echo "${WIDTH},${HEIGHT},${TB_PER_SEC}" >> "$OUTPUT_FILE"
done
