#!/bin/bash
#
# Combined Parallel Test for BeeGFS Using dd and fio
#
# This script runs a set of tests concurrently where each test uses:
#   - A specified File Size (in MB)
#   - A BeeGFS Chunk Size (in KB, configured on a per-directory basis)
#   - An I/O Block Size (for dd and fio, e.g. "256K", "512K", "1M", "2M")
#
# For each combination the following steps are performed concurrently:
#   1. The proper test directory is created (if not already present) and its
#      BeeGFS chunk pattern is set.
#   2. A test file is created (or overwritten) to the specified file size.
#   3. A sequential dd write test runs (measuring time and throughput).
#   4. A sequential dd read test runs.
#   5. FIO sequential write and read tests run with the same parameters,
#      extracting the reported bandwidth.
#   6. The results are immediately appended to a text file using a lock.
#
# The script spawns background jobs for each test combination and limits the
# number of concurrent jobs to maximize resource usage on a VM with 16 cores.
#

# -------------------------------
# Global Directories and Result File
# -------------------------------
BASE_DIR="/mnt/beegfs"
LOG_DIR="${BASE_DIR}/test_logs"
RESULT_FILE="${LOG_DIR}/combined_parallel_results.txt"
mkdir -p "${LOG_DIR}"

# Write header for the result file.
cat <<EOF > "${RESULT_FILE}"
Combined Parallel Test Results for BeeGFS (dd and fio)
Generated on: $(date)
------------------------------------------------------------
EOF

# -------------------------------
# Parameter Arrays
# -------------------------------
# File sizes in MB.
FILE_SIZES=(100 200 500 1000 5000 10000 20000)
# BeeGFS chunk sizes in KB (each combination uses its own directory).
CHUNK_SIZES=(64 128 256 512 768 1024 2048 4096)
# I/O block sizes (strings acceptable by dd and fio).
BS_ARRAY=("256K" "512K" "1M" "5M" "10M" "50M" "100M")

# -------------------------------
# Pre-create Test Directories
# -------------------------------
# For each BeeGFS chunk size, create (if needed) a test directory and configure it.
for chunk in "${CHUNK_SIZES[@]}"; do
    test_dir="${BASE_DIR}/testdir_${chunk}K"
    if [ ! -d "${test_dir}" ]; then
        echo "Creating directory: ${test_dir}"
        mkdir -p "${test_dir}"
        sudo beegfs-ctl --setpattern --chunksize="${chunk}k" "${test_dir}"
        echo "Configured ${test_dir} with chunk size ${chunk}K"
    fi
done

# -------------------------------
# Helper Functions
# -------------------------------

# Convert a block size string (e.g. "256K", "1M") to a number of bytes.
convert_bs_to_bytes() {
    local bs_str=$1
    if [[ ${bs_str} == *K ]]; then
        echo $(( ${bs_str%K} * 1024 ))
    elif [[ ${bs_str} == *M ]]; then
        echo $(( ${bs_str%M} * 1024 * 1024 ))
    else
        echo "${bs_str}"  # Assume already in bytes.
    fi
}

# Run a dd test (sequential write or read) and return the elapsed time.
#   $1: "write" or "read"
#   $2: file path
#   $3: block size (e.g. "256K")
#   $4: count (only used for write)
run_dd_test() {
    local direction=$1
    local file_path=$2
    local bs=$3
    local count=$4
    local start_time end_time dt

    if [ "${direction}" == "write" ]; then
        start_time=$(date +%s.%N)
        sudo dd if=/dev/zero of="${file_path}" bs="${bs}" count="${count}" conv=fdatasync status=none
        end_time=$(date +%s.%N)
    elif [ "${direction}" == "read" ]; then
        start_time=$(date +%s.%N)
        sudo dd if="${file_path}" of=/dev/null bs="${bs}" status=none
        end_time=$(date +%s.%N)
    fi
    dt=$(echo "$end_time - $start_time" | bc -l)
    echo "$dt"
}

# Run an fio test for a given mode ("read" or "write") and extract the bandwidth.
#   $1: mode ("read" or "write")
#   $2: file path
#   $3: block size (e.g. "256K")
#   $4: file size in MB (for the fio --size parameter)
run_fio_test() {
    local mode=$1
    local file_path=$2
    local bs=$3
    local fsize_mb=$4
    local job_file="/tmp/fio_job_$$.job"

    cat > "$job_file" <<EOF
[global]
ioengine=libaio
direct=1
overwrite=1
filename=${file_path}
bs=${bs}
size=${fsize_mb}M
numjobs=1
group_reporting

[${mode}_job]
rw=${mode}
EOF

    local fio_output
    fio_output=$(fio --output-format=normal "$job_file")
    rm -f "$job_file"

    local bw
    bw=$(echo "$fio_output" | grep -i "^ *${mode}:" | grep -o "BW=[^,]*" | head -n 1 | cut -d= -f2)
    [ -z "$bw" ] && bw="N/A"
    echo "$bw"
}

# -------------------------------
# Test Case Function
# -------------------------------
# Runs dd and fio tests for a given file size (MB), chunk size (KB), and block size.
# Appends results to the global result file.
test_case() {
    local fsize=$1
    local chunk=$2
    local bs=$3

    local test_dir="${BASE_DIR}/testdir_${chunk}K"
    local test_file="${test_dir}/test_${fsize}M_${bs}.dat"

    # Calculate file size in bytes and number of dd blocks.
    local file_size_bytes=$(( fsize * 1024 * 1024 ))
    local bs_bytes
    bs_bytes=$(convert_bs_to_bytes "${bs}")
    local count=$(( file_size_bytes / bs_bytes ))

    # Remove any previous test file.
    [ -f "${test_file}" ] && rm -f "${test_file}"

    # --- dd Write Test ---
    local dd_write_time
    dd_write_time=$(run_dd_test "write" "${test_file}" "${bs}" "${count}")
    local dd_write_bw
    dd_write_bw=$(echo "scale=2; ${fsize} / ${dd_write_time}" | bc -l)

    # --- dd Read Test ---
    local dd_read_time
    dd_read_time=$(run_dd_test "read" "${test_file}" "${bs}")
    local dd_read_bw
    dd_read_bw=$(echo "scale=2; ${fsize} / ${dd_read_time}" | bc -l)

    # --- fio Tests ---
    local fio_write_bw
    fio_write_bw=$(run_fio_test "write" "${test_file}" "${bs}" "${fsize}")
    local fio_read_bw
    fio_read_bw=$(run_fio_test "read" "${test_file}" "${bs}" "${fsize}")

    # Compose the result output.
    local result_output
    result_output=$(cat <<EOF
Test Combination:
  File Size    = ${fsize} MB
  Chunk Size   = ${chunk} KB
  Block Size   = ${bs}
Results:
  DD Write : Time = ${dd_write_time} sec, Throughput = ${dd_write_bw} MB/s
  DD Read  : Time = ${dd_read_time} sec, Throughput = ${dd_read_bw} MB/s
  FIO Write: Bandwidth = ${fio_write_bw}
  FIO Read : Bandwidth = ${fio_read_bw}
------------------------------------------------------------
EOF
)

    # Append the result safely using a file lock.
    {
      flock -x 200
      echo "${result_output}" >> "${RESULT_FILE}"
    } 200>"${RESULT_FILE}.lock"

    # Remove test file to free up disk space.
    rm -f "${test_file}"
}

# -------------------------------
# Main Loop: Launch Tests in Parallel
# -------------------------------
# Set a maximum number of concurrent jobs.
max_jobs=16

for fsize in "${FILE_SIZES[@]}"; do
  for chunk in "${CHUNK_SIZES[@]}"; do
    for bs in "${BS_ARRAY[@]}"; do
      # Launch the test case as a background job.
      test_case "$fsize" "$chunk" "$bs" &
      
      # Limit the number of concurrent jobs.
      while [ "$(jobs -rp | wc -l)" -ge "$max_jobs" ]; do
          sleep 1
      done
    done
  done
done

# Wait for all background jobs to finish.
wait

echo "All tests completed. Results are available in: ${RESULT_FILE}"