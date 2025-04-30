#!/bin/bash
#
# BeeGFS Parallel Test Script – Multiple Tests Per File Using 16 Cores
#


BASE_DIR="/mnt/beegfs"
LOG_DIR="${BASE_DIR}/test_logs"
CSV_FILE="${LOG_DIR}/test_results.csv"

mkdir -p "${LOG_DIR}"

if [ ! -f "${CSV_FILE}" ]; then
    echo "file_path,file_size_KB,chunk_size_KB,access_count,avg_read_KB,avg_write_KB,max_read_KB,max_write_KB,read_ops,write_ops,throughput_KBps,error_message" > "${CSV_FILE}"
fi

# Parameters
CHUNK_SIZES=(64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072)  #(64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072)  # in KB
FILE_SIZES=(1024) #(1024 2048 5120 10240 20480 51200 102400 204800 512000 1024000 5120000 10240000 20480000 51200000)  # in KB (i.e., 100MB=102400KB, 200MB=204800KB, etc.)
READ_OPS=(5 10 20)
WRITE_OPS=(2 5 10)

iterations=5       # Number of full test cycles
tests_per_file=200     # Random tests to perform per file

echo "Starting test with $iterations iterations, $tests_per_file tests per file..."
echo "------------------------------------------------------------"

for ((iter=1; iter<=iterations; iter++)); do
    fs=${FILE_SIZES[$(( RANDOM % ${#FILE_SIZES[@]} ))]}
    file_path=$(mktemp "${BASE_DIR}/testfile_${iter}_XXXXXX.dat")

    echo "[$iter/$iterations] Creating file ${file_path} of size ${fs}KB..."
    sudo dd if=/dev/zero of="${file_path}" bs=1K count="${fs}" status=none

    if [ $? -ne 0 ] || [ ! -f "${file_path}" ]; then
        echo "❌ Failed to create file ${file_path}"
        continue
    fi

    for ((t=1; t<=tests_per_file; t++)); do
        cs=${CHUNK_SIZES[$(( RANDOM % ${#CHUNK_SIZES[@]} ))]}
        ro=${READ_OPS[$(( RANDOM % ${#READ_OPS[@]} ))]}
        wo=${WRITE_OPS[$(( RANDOM % ${#WRITE_OPS[@]} ))]}
        error_msg=""
        total_read_kb=0
        max_read_kb=0

        for ((j=0; j<ro; j++)); do
            max_blk=$(( cs > 512 ? 512 : cs ))
            rsize=$(( RANDOM % max_blk + 4 ))
            max_off=$(( fs - rsize ))
            [ $max_off -le 0 ] && max_off=1
            off=$(( RANDOM % max_off ))
            sudo dd if="${file_path}" of=/dev/null bs=1K skip="${off}" count="${rsize}" status=none 2>/dev/null
            [ $? -ne 0 ] && error_msg="Error during read" && break
            total_read_kb=$(( total_read_kb + rsize ))
            [ $rsize -gt $max_read_kb ] && max_read_kb=$rsize
        done
        [ $ro -gt 0 ] && avg_read_KB=$(echo "scale=2; $total_read_kb / $ro" | bc) || avg_read_KB=0

        total_write_kb=0
        max_write_kb=0
        for ((j=0; j<wo; j++)); do
            max_blk=$(( cs > 512 ? 512 : cs ))
            wsize=$(( RANDOM % max_blk + 4 ))
            max_off=$(( fs - wsize ))
            [ $max_off -le 0 ] && max_off=1
            off=$(( RANDOM % max_off ))
            sudo dd if=/dev/urandom of="/tmp/random_data_$$" bs=1K count="${wsize}" status=none 2>/dev/null
            [ $? -ne 0 ] && error_msg="Error generating data" && break
            sudo dd if="/tmp/random_data_$$" of="${file_path}" bs=1K seek="${off}" count="${wsize}" conv=notrunc status=none 2>/dev/null
            [ $? -ne 0 ] && error_msg="Error writing to file" && rm -f "/tmp/random_data_$$" && break
            rm -f "/tmp/random_data_$$"
            total_write_kb=$(( total_write_kb + wsize ))
            [ $wsize -gt $max_write_kb ] && max_write_kb=$wsize
        done
        [ $wo -gt 0 ] && avg_write_KB=$(echo "scale=2; $total_write_kb / $wo" | bc) || avg_write_KB=0

        # Throughput
        sample_KB=$fs
        [ $fs -gt 102400 ] && sample_KB=102400
        start=$(date +%s.%N)
        sudo dd if="${file_path}" of=/dev/null bs=1K count="${sample_KB}" status=none 2>/dev/null
        [ $? -ne 0 ] && error_msg="Error measuring throughput"
        end=$(date +%s.%N)
        dur=$(echo "$end - $start" | bc)
        throughput_KBps=$(echo "scale=2; $sample_KB / $dur" | bc)

        access_count=$(( ro + wo ))
        { flock -x 200; echo "${file_path},${fs},${cs},${access_count},${avg_read_KB},${avg_write_KB},${max_read_kb},${max_write_kb},${ro},${wo},${throughput_KBps},${error_msg}" >> "${CSV_FILE}"; } 200>"${CSV_FILE}.lock"

        echo "✅ Test $t logged for ${file_path} — Chunk=${cs}KB, Throughput=${throughput_KBps} KB/s"
    done

    echo "🧹 Deleting file: ${file_path}"
    rm -f "${file_path}"
done

echo "✅ All tests complete. Results saved to: ${CSV_FILE}"





# #!/bin/bash
# #
# # BeeGFS Parallel Test Script – Multiple Tests Per File Using 16 Cores
# #

# BASE_DIR="/mnt/beegfs"
# LOG_DIR="${BASE_DIR}/test_logs"
# CSV_FILE="${LOG_DIR}/test_results.csv"
# MAX_PARALLEL=1

# mkdir -p "${LOG_DIR}"

# if [ ! -f "${CSV_FILE}" ]; then
#     echo "file_path,file_size_KB,chunk_size_KB,access_count,avg_read_KB,avg_write_KB,max_read_KB,max_write_KB,read_ops,write_ops,throughput_KBps,error_message" > "${CSV_FILE}"
# fi

# CHUNK_SIZES=(64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072)
# FILE_SIZES=(1024) # (1024 2048 5120 10240 20480 51200 102400 204800 512000 1024000 5120000 10240000 20480000 51200000)
# READ_OPS=(5 10 20)
# WRITE_OPS=(2 5 10)

# iterations=5
# tests_per_file=200

# echo "🚀 Starting test with $iterations iterations, $tests_per_file tests per file using $MAX_PARALLEL cores..."
# echo "------------------------------------------------------------"

# # Function for single test run
# run_test() {
#     local file_path=$1
#     local fs=$2
#     local t=$3
#     local cs=${CHUNK_SIZES[$(( RANDOM % ${#CHUNK_SIZES[@]} ))]}
#     local ro=${READ_OPS[$(( RANDOM % ${#READ_OPS[@]} ))]}
#     local wo=${WRITE_OPS[$(( RANDOM % ${#WRITE_OPS[@]} ))]}
#     local error_msg=""
#     local total_read_kb=0
#     local max_read_kb=0

#     for ((j=0; j<ro; j++)); do
#         rsize=$(( RANDOM % 512 + 4 ))
#         max_off=$(( fs - rsize ))
#         [ $max_off -le 0 ] && max_off=1
#         off=$(( RANDOM % max_off ))
#         sudo dd if="${file_path}" of=/dev/null bs=1K skip="${off}" count="${rsize}" status=none 2>/dev/null
#         [ $? -ne 0 ] && error_msg="Error during read" && break
#         total_read_kb=$(( total_read_kb + rsize ))
#         [ $rsize -gt $max_read_kb ] && max_read_kb=$rsize
#     done
#     [ $ro -gt 0 ] && avg_read_KB=$(echo "scale=2; $total_read_kb / $ro" | bc) || avg_read_KB=0

#     total_write_kb=0
#     max_write_kb=0
#     for ((j=0; j<wo; j++)); do
#         wsize=$(( RANDOM % 512 + 4 ))
#         max_off=$(( fs - wsize ))
#         [ $max_off -le 0 ] && max_off=1
#         off=$(( RANDOM % max_off ))
#         dd if=/dev/urandom of="/tmp/random_data_$$" bs=1K count="${wsize}" status=none
#         [ $? -ne 0 ] && error_msg="Error generating data" && break
#         sudo dd if="/tmp/random_data_$$" of="${file_path}" bs=1K seek="${off}" count="${wsize}" conv=notrunc status=none 2>/dev/null
#         [ $? -ne 0 ] && error_msg="Error writing to file" && rm -f "/tmp/random_data_$$" && break
#         rm -f "/tmp/random_data_$$"
#         total_write_kb=$(( total_write_kb + wsize ))
#         [ $wsize -gt $max_write_kb ] && max_write_kb=$wsize
#     done
#     [ $wo -gt 0 ] && avg_write_KB=$(echo "scale=2; $total_write_kb / $wo" | bc) || avg_write_KB=0

#     sample_KB=$fs; [ $fs -gt 102400 ] && sample_KB=102400
#     start=$(date +%s.%N)
#     sudo dd if="${file_path}" of=/dev/null bs=1K count="${sample_KB}" status=none 2>/dev/null
#     end=$(date +%s.%N)
#     dur=$(echo "$end - $start" | bc)
#     throughput_KBps=$(echo "scale=2; $sample_KB / $dur" | bc)

#     access_count=$(( ro + wo ))

#     { flock -x 200
#       echo "${file_path},${fs},${cs},${access_count},${avg_read_KB},${avg_write_KB},${max_read_kb},${max_write_kb},${ro},${wo},${throughput_KBps},${error_msg}" >> "${CSV_FILE}"
#     } 200>"${CSV_FILE}.lock"

#     echo "✅ Test $t logged for ${file_path} — Chunk=${cs}KB, Thpt=${throughput_KBps} KB/s"
# }

# # Main loop for each file
# for ((iter=1; iter<=iterations; iter++)); do
#     fs=${FILE_SIZES[$(( RANDOM % ${#FILE_SIZES[@]} ))]}
#     file_path=$(mktemp "${BASE_DIR}/testfile_${iter}_XXXXXX.dat")
#     echo "[$iter/$iterations] Creating file ${file_path} of size ${fs}KB..."
#     sudo dd if=/dev/zero of="${file_path}" bs=1K count="${fs}" status=none

#     if [ $? -ne 0 ] || [ ! -f "${file_path}" ]; then
#         echo "❌ Failed to create file ${file_path}"
#         continue
#     fi

#     job_count=0
#     for ((t=1; t<=tests_per_file; t++)); do
#         run_test "${file_path}" "${fs}" "${t}" &
#         ((job_count++))
#         if (( job_count >= MAX_PARALLEL )); then
#             wait -n
#             ((job_count--))
#         fi
#     done

#     wait  # Wait for all test jobs for this file
#     echo "🧹 Deleting file: ${file_path}"
#     rm -f "${file_path}"
# done

# echo "✅ All parallel file tests complete. Results saved to: ${CSV_FILE}"




#!/bin/bash
#
# BeeGFS Testing Script – One File, Multiple Tests Per Iteration
#
