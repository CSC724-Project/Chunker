#!/bin/bash

# BeeGFS Expanded Testing Script with Error Echos

BASE_DIR="/mnt/beegfs"
LOG_DIR="${BASE_DIR}/test_logs"
CSV_FILE="${LOG_DIR}/beegfs_test_results.csv"

mkdir -p "${LOG_DIR}" || echo "ERROR: Failed to create log directory at ${LOG_DIR}"

if [ ! -f "${CSV_FILE}" ]; then
    echo "file_path,file_size,chunk_size,access_count,avg_read_size,avg_write_size,max_read_size,max_write_size,read_count,write_count,throughput_mbps" > "${CSV_FILE}" || echo "ERROR: Failed to initialize CSV file at ${CSV_FILE}"
fi

CHUNK_SIZES=(64 128 256 512 768 1024 1536 2048 3072 4096 6144 8192)
FILE_SIZES=(50 100 200 350 500 750 1000 1500 2000)
READ_PATTERNS=(5 10 15 20 25 30 40 50)
WRITE_PATTERNS=(2 5 8 10 15 20 25 30)

echo "Creating and configuring test directories..."
for chunk_size in "${CHUNK_SIZES[@]}"; do
    dir_name="${BASE_DIR}/testdir_${chunk_size}K"
    if [ ! -d "${dir_name}" ]; then
        echo "Creating directory for ${chunk_size}K chunk size: ${dir_name}"
        mkdir -p "${dir_name}" || echo "ERROR: Failed to create directory ${dir_name}"
        
        echo "Setting chunk size to ${chunk_size}K for ${dir_name}"
        sudo beegfs-ctl --setpattern --chunksize="${chunk_size}k" "${dir_name}" || echo "ERROR: Failed to set chunk size for ${dir_name}"
    fi
done

random_range() {
    local min=$1
    local max=$2
    echo $((RANDOM % (max - min + 1) + min))
}

run_test() {
    local chunk_size=$1
    local file_size=$2
    local read_count=$3
    local write_count=$4
    
    local dir_path="${BASE_DIR}/testdir_${chunk_size}K"
    local file_name="test_${file_size}M.dat"
    local file_path="${dir_path}/${file_name}"
    
    echo "========================================================"
    echo "Running test: Chunk Size ${chunk_size}KB, File Size ${file_size}MB"
    echo "Read count: ${read_count}, Write count: ${write_count}"
    echo "========================================================"
    
    if [ ! -f "${file_path}" ]; then
        echo "Creating test file: ${file_path} (${file_size}MB)"
        sudo dd if=/dev/zero of="${file_path}" bs=1M count="${file_size}" status=progress || echo "ERROR: Failed to create test file ${file_path}"
    else
        echo "Using existing file: ${file_path} (${file_size}MB)"
    fi
    
    if [ ! -f "${file_path}" ]; then
        echo "ERROR: Test file does not exist after creation attempt: ${file_path}"
        return 1
    fi

    echo "Simulating ${read_count} random reads"
    local total_read_size=0
    local max_read_size=0
    
    for ((i=1; i<=read_count; i++)); do
        local max_read_block=$((chunk_size > 512 ? 512 : chunk_size))
        local read_size=$((RANDOM % max_read_block + 4))
        local read_size_bytes=$((read_size * 1024))
        local max_offset=$((file_size * 1024 * 1024 - read_size_bytes))
        [ ${max_offset} -le 0 ] && max_offset=1
        local offset=$((RANDOM % max_offset))
        
        dd if="${file_path}" of=/dev/null bs=1K skip=$((offset / 1024)) count=${read_size} status=none 2>/dev/null || echo "ERROR: Failed read operation at offset $offset"
        
        total_read_size=$((total_read_size + read_size_bytes))
        [ ${read_size_bytes} -gt ${max_read_size} ] && max_read_size=${read_size_bytes}
    done
    
    local avg_read_size=0
    [ ${read_count} -gt 0 ] && avg_read_size=$((total_read_size / read_count))

    echo "Simulating ${write_count} random writes"
    local total_write_size=0
    local max_write_size=0
    
    for ((i=1; i<=write_count; i++)); do
        local max_write_block=$((chunk_size > 512 ? 512 : chunk_size))
        local write_size=$((RANDOM % max_write_block + 4))
        local write_size_bytes=$((write_size * 1024))
        local max_offset=$((file_size * 1024 * 1024 - write_size_bytes))
        [ ${max_offset} -le 0 ] && max_offset=1
        local offset=$((RANDOM % max_offset))
        
        dd if=/dev/urandom of="/tmp/random_data_$$" bs=1K count=${write_size} status=none 2>/dev/null || echo "ERROR: Failed to generate random data"
        
        dd if="/tmp/random_data_$$" of="${file_path}" bs=1K seek=$((offset / 1024)) count=${write_size} conv=notrunc status=none 2>/dev/null || echo "ERROR: Failed write operation at offset $offset"
        
        rm -f "/tmp/random_data_$$" || echo "ERROR: Failed to remove temporary random data file"
        
        total_write_size=$((total_write_size + write_size_bytes))
        [ ${write_size_bytes} -gt ${max_write_size} ] && max_write_size=${write_size_bytes}
    done
    
    local avg_write_size=0
    [ ${write_count} -gt 0 ] && avg_write_size=$((total_write_size / write_count))

    echo "Measuring throughput"
    local sample_size=$((file_size > 100 ? 100 : file_size))
    start_time=$(date +%s.%N)
    dd if="${file_path}" of=/dev/null bs=1M count=${sample_size} status=none 2>/dev/null || echo "ERROR: Failed throughput measurement"
    end_time=$(date +%s.%N)
    
    local duration=$(echo "${end_time} - ${start_time}" | bc)
    local throughput=$(echo "scale=2; ${sample_size} / ${duration}" | bc)
    
    local access_count=$((read_count + write_count))
    
    echo "${file_path},${file_size}000000,${chunk_size}000,${access_count},${avg_read_size},${avg_write_size},${max_read_size},${max_write_size},${read_count},${write_count},${throughput}" >> "${CSV_FILE}" || echo "ERROR: Failed to write results to CSV"

    echo "Test completed and logged to CSV"
}

generate_random_combinations() {
    local num_combinations=$1
    
    echo "Generating ${num_combinations} random test combinations..."
    
    for ((i=1; i<=num_combinations; i++)); do
        local chunk_idx=$((RANDOM % ${#CHUNK_SIZES[@]}))
        local file_idx=$((RANDOM % ${#FILE_SIZES[@]}))
        local read_idx=$((RANDOM % ${#READ_PATTERNS[@]}))
        local write_idx=$((RANDOM % ${#WRITE_PATTERNS[@]}))
        
        local chunk_size=${CHUNK_SIZES[$chunk_idx]}
        local file_size=${FILE_SIZES[$file_idx]}
        local read_count=${READ_PATTERNS[$read_idx]}
        local write_count=${WRITE_PATTERNS[$write_idx]}
        
        echo "Running combination $i of $num_combinations: Chunk ${chunk_size}K, File ${file_size}MB, Reads ${read_count}, Writes ${write_count}"
        
        run_test "$chunk_size" "$file_size" "$read_count" "$write_count"
        
        echo "Progress: $i/$num_combinations ($(echo "scale=2; $i*100/$num_combinations" | bc)%)"
    done
}

echo "Starting BeeGFS expanded testing to generate 1500 data points"
echo "Results will be saved to: ${CSV_FILE}"

generate_random_combinations 50

echo "Testing completed. Results are available in: ${CSV_FILE}"
echo "To clean up test files, run: $(basename "$0") --cleanup"

if [ "$1" == "--cleanup" ]; then
    echo "Cleaning up test files..."
    for chunk_size in "${CHUNK_SIZES[@]}"; do
        dir_path="${BASE_DIR}/testdir_${chunk_size}K"
        if [ -d "${dir_path}" ]; then
            echo "Removing test directory: ${dir_path}"
            rm -rf "${dir_path}" || echo "ERROR: Failed to remove directory ${dir_path}"
        fi
    done
    echo "Keeping log directory: ${LOG_DIR}"
fi