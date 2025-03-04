#!/bin/bash
#SBATCH --job-name=zeiss_compression   # Job name
#SBATCH --nodes=5                      # Number of nodes
#SBATCH --ntasks=5                     # One task per node
#SBATCH --cpus-per-task=4              # Adjust CPU cores per task
#SBATCH --time=02:00:00                # Max runtime (hh:mm:ss)
#SBATCH --output=zeiss_compression_%j.out  # Standard output log
#SBATCH --error=zeiss_compression_%j.err   # Standard error log
#SBATCH --partition=PARTITION              # Use appropriate partition
#SBATCH --mail-type=END,FAIL           # Notifications for job done & fail
#SBATCH --mail-user=your_email@example.com # Change to your email

module load python                     # Load Python if needed
source ~/your_virtual_env/bin/activate # Activate virtual environment if needed

# Launch jobs with different partition IDs
srun --nodes=1 --ntasks=1 python example.py 0 &
srun --nodes=1 --ntasks=1 python example.py 1 &
srun --nodes=1 --ntasks=1 python example.py 2 &
srun --nodes=1 --ntasks=1 python example.py 3 &
srun --nodes=1 --ntasks=1 python example.py 4 &

wait  # Ensure all processes complete