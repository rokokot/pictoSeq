#!/bin/bash -l
#SBATCH --clusters=genius
#SBATCH --account=intro_vsc37132
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --partition=gpu_p100
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=5000M
#SBATCH --job-name=propicto_p100
#SBATCH --output=logs/propicto_p100_%j.out
#SBATCH --error=logs/propicto_p100_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting ProPicto Research on P100 GPU"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPU: $SLURM_GPUS_PER_NODE P100"
echo "Time: $(date)"
echo ""

# Load required modules
echo " Loading modules..."
module purge
module load miniconda3

# Activate conda environment
echo " Activating pictoseq environment..."
source activate pictoseq

# Check CUDA availability
echo " Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# Set up environment
echo " Setting up paths..."
cd $VSC_DATA/pictoSeq

# Check data availability
echo " Checking data availability..."
ls -la data/processed_propicto/

echo ""
echo " Starting ProPicto experiments..."
echo "Will save results to: $VSC_SCRATCH/pictoSeq_results"
echo ""

# Run the main research pipeline
python main_runner.py \
    --run-all \
    --results-path "$VSC_SCRATCH/pictoSeq_results" \
    --max-train 50000 \
    --max-val 5000 \
    --max-test 5000 \
    --epochs 5

# Check if job completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "ProPicto research completed successfully!"
    echo " Results saved to: $VSC_SCRATCH/pictoSeq_results"
    echo " Completed at: $(date)"
else
    echo ""
    echo " experiment failed!"
    echo "  at: $(date)"
    exit 1
fi

echo ""
echo " Summary:"
echo "   Job ID: $SLURM_JOB_ID"
echo "   Node: $SLURMD_NODENAME" 
echo "   GPU: P100"
echo "   Duration: $(date)"
echo "   Results: $VSC_SCRATCH/pictoSeq_results"
echo ""
echo " finished!"