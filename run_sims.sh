#!/bin/bash
# run_parallel.sh
SCRIPT="my_script.py"       # your Python script
TOTAL_RUNS=100              # total number of runs
MAX_PROCS=5                 # max parallel jobs at a time

# Loop over the total runs
for ((i=1; i<=TOTAL_RUNS; i++)); do
    echo "Starting run $i..."
    python $SCRIPT --n "$i" &   # add more args if needed

    # Wait if MAX_PROCS are running
    while (( $(jobs -r | wc -l) >= MAX_PROCS )); do
        sleep 1
    done
done

# Wait for any remaining jobs to finish
wait

echo "All $TOTAL_RUNS runs completed."