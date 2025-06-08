#!/bin/bash
sbatch <<EOT
#!/bin/bash -l
#SBATCH --job-name sssp-tests
#SBATCH --output output.txt
#SBATCH --account "g101-2284"
#SBATCH --nodes $1
#SBATCH --ntasks-per-node $2
#SBATCH --time 00:10:00

module load common/python/3.11
python3 run_test_perf.py $1 $2 ${3}
EOT
