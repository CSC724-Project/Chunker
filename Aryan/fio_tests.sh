#!/bin/bash
#
# This script generates and runs 250 FIO test cases using different
# combinations of runtime, block size, file size, I/O mode, and number of threads.
# Each test creates a file with a known name; after the test, the script queries 
# BeeGFS to obtain the file entry information (which includes stripe/chunk details).
#
# The output for each test is stored in an individual .txt file and a master summary
# file ("fio_summary.txt") is maintained for a quick review.
#
# Usage:
#   chmod +x run_fio_tests.sh
#   sudo ./run_fio_tests.sh
#
# NOTE: Ensure FIO is installed, the BeeGFS mount point (/mnt/beegfs) is valid,
#       and beegfs-ctl is configured (or the client config file is in its default
#       location, /etc/beegfs/beegfs-client.conf).

# Define parameter arrays (adjust these values as needed)
runtimes=(10 15)                          # in seconds
bss=("256K" "512K" "1M" "2M")                 # block sizes
file_sizes=("256M" "512M" "1G" "5G" "10G" "20G")             # total file sizes for the test file
rw_modes=("randread" "randwrite" "randrw")     # I/O modes: random read, random write, and mixed
threads=(1 2 4 8 16)                            # number of threads (numjobs)

# Create a directory for FIO results
result_dir="./fio_results"
mkdir -p "$result_dir"

# Master summary file to accumulate human-readable results
master_results_file="${result_dir}/fio_summary.txt"
echo "FIO Test Summary" > "$master_results_file"
echo "================" >> "$master_results_file"

# Initialize test case counter and maximum tests to run
test_case=0
max_tests=5

# Loop over parameter combinations until reaching max_tests
for runtime in "${runtimes[@]}"; do
  for bs in "${bss[@]}"; do
    for size in "${file_sizes[@]}"; do
      for rw in "${rw_modes[@]}"; do
        for thread in "${threads[@]}"; do
          test_case=$((test_case + 1))
          # Stop if we've reached the maximum number of test cases
          if [ "$test_case" -gt "$max_tests" ]; then
            break 5
          fi

          # Define a temporary job file for this test case
          job_file="/tmp/beegfs_fio_job_${test_case}.job"
          # Define the filename that FIO will use for this test
          fio_filename="/mnt/beegfs/fio_test_${test_case}.dat"

          # Create a new FIO job file with the current parameters
          cat > "$job_file" <<EOF
[global]
ioengine=libaio
direct=1
runtime=${runtime}
time_based
directory=/mnt/beegfs
bs=${bs}
size=${size}
numjobs=${thread}
filename=${fio_filename}

[${rw}-job]
rw=${rw}
EOF

          # Define an output file for FIO results (with a .txt extension)
          output_file="${result_dir}/fio_result_${test_case}.txt"

          echo "Running test case ${test_case}: runtime=${runtime}s, bs=${bs}, size=${size}, rw=${rw}, threads=${thread}"
          # Run FIO with the generated job file and save its output
          fio "$job_file" > "$output_file" 2>&1

          # Query BeeGFS file entry info using the additional parameters for mount point and verbose output.
          beegfs_info=$(sudo beegfs-ctl --getentryinfo --mount=/mnt/beegfs/"${fio_filename}" --verbose 2>&1)

          # Append a summary of this test to the master summary file for easy review
          {
            echo "----------------------------------------"
            echo "Test Case ${test_case}:"
            echo "Parameters: runtime=${runtime}s, bs=${bs}, size=${size}, rw=${rw}, threads=${thread}"
            echo "FIO Output File: ${output_file}"
            echo ""
            echo "---------- FIO Output Snapshot ----------"
            head -n 20 "$output_file"
            echo ""
            echo "---------- BeeGFS Entry Info (Stripe/Chunk, etc.) ----------"
            echo "${beegfs_info}"
            echo "----------------------------------------"
            echo ""
          } >> "$master_results_file"

          # Clean up: Remove the temporary job file and the FIO data file
          rm -f "$job_file"
          rm -f "${fio_filename}"
        done
      done
    done
  done
done

echo "Completed ${test_case} test cases."
echo "Check the ${result_dir} directory for detailed results and summary in fio_summary.txt."


#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------


# #!/bin/bash
# #
# # This script generates and runs 250 FIO test cases using different
# # combinations of runtime, block size, file size, I/O mode, and number of threads.
# # Each test result is stored in a separate text file, and a summary of all tests
# # is appended to a master summary file called "fio_summary.txt".
# #
# # Usage:
# #   chmod +x run_fio_tests.sh
# #   sudo ./run_fio_tests.sh
# #
# # NOTE: Ensure FIO is installed and the BeeGFS mount point (/mnt/beegfs) is valid.

# # Define parameter arrays (adjust these values as needed)
# runtimes=(30 60 90)                    # in seconds
# bss=("256K" "512K" "1M" "2M")           # block sizes
# file_sizes=("1G" "5G" "10G" "20G")       # total file sizes for the test file
# rw_modes=("randread" "randwrite" "randrw")  # I/O modes: random read, random write, and mixed
# threads=(1 2 4 8)                      # number of threads (numjobs)

# # Create a directory for FIO results
# result_dir="./fio_results"
# mkdir -p "$result_dir"

# # Master summary file to accumulate human-readable results
# master_results_file="${result_dir}/fio_summary.txt"
# echo "FIO Test Summary" > "$master_results_file"
# echo "================" >> "$master_results_file"

# # Initialize test case counter and maximum tests to run
# test_case=0
# max_tests=250

# # Loop over parameter combinations until reaching max_tests
# for runtime in "${runtimes[@]}"; do
#   for bs in "${bss[@]}"; do
#     for size in "${file_sizes[@]}"; do
#       for rw in "${rw_modes[@]}"; do
#         for thread in "${threads[@]}"; do
#           test_case=$((test_case + 1))
#           # Stop if we've reached the maximum number of test cases
#           if [ "$test_case" -gt "$max_tests" ]; then
#             break 5
#           fi

#           # Define a temporary job file for this test case
#           job_file="/tmp/beegfs_fio_job_${test_case}.job"

#           # Create a new FIO job file with the current parameters
#           cat > "$job_file" <<EOF
# [global]
# ioengine=libaio
# direct=1
# runtime=${runtime}
# time_based
# directory=/mnt/beegfs
# bs=${bs}
# size=${size}
# numjobs=${thread}

# [${rw}-job]
# rw=${rw}
# EOF

#           # Define an output file for FIO results (with a .txt extension)
#           output_file="${result_dir}/fio_result_${test_case}.txt"

#           echo "Running test case ${test_case}: runtime=${runtime}s, bs=${bs}, size=${size}, rw=${rw}, threads=${thread}"
#           # Run FIO with the generated job file and save its output
#           fio "$job_file" > "$output_file" 2>&1

#           # Append a summary of this test to the master summary file for easy review
#           {
#             echo "----------------------------------------"
#             echo "Test Case ${test_case}:"
#             echo "Parameters: runtime=${runtime}s, bs=${bs}, size=${size}, rw=${rw}, threads=${thread}"
#             echo "Output File: ${output_file}"
#             echo "---------- Output Snapshot ----------"
#             head -n 20 "$output_file"
#             echo "----------------------------------------"
#             echo ""
#           } >> "$master_results_file"

#           # Remove the temporary job file after running the test
#           rm -f "$job_file"
#         done
#       done
#     done
#   done
# done

# echo "Completed ${test_case} test cases."
# echo "Check the ${result_dir} directory for detailed results and summary in fio_summary.txt."
