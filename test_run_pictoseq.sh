#!/bin/bash -l
#SBATCH --clusters=genius
#SBATCH --account=intro_vsc37132
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --partition=gpu_p100
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=5000M
#SBATCH --job-name=pictoseq_test_p100
#SBATCH --output=logs/propicto_test_p100_%j.out
#SBATCH --error=logs/propicto_test_p100_%j.err

mkdir -p logs

echo "PictoSeq Test Run on P100 GPU"
echo "================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPU: $SLURM_GPUS_PER_NODE P100"
echo "Time: $(date)"
echo ""

echo "Loading modules..."
module purge
module load miniconda3

# Activate conda environment
echo "🐍 Activating pictoseq environment..."
source activate pictoseq

# Check CUDA availability
echo "🔍 Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"

# Set up environment
echo "📁 Setting up paths..."
cd $VSC_DATA/pictoSeq

# Check data availability
echo "📊 Checking data availability..."
ls -la data/processed_propicto/

echo ""
echo "🧪 Starting ProPicto TEST RUN..."
echo "Will save results to: $VSC_SCRATCH/pictoSeq_results"
echo "TEST MODE: Limited samples and epochs for validation"
echo ""

# Run the test version
python main_runner.py \
    --run-all \
    --test-run \
    --results-path "$VSC_SCRATCH/pictoSeq_results" \
    --max-train 1000 \
    --max-val 200 \
    --max-test 200 \
    --epochs 2

# Check if job completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ ProPicto TEST completed successfully!"
    echo "📁 Results saved to: $VSC_SCRATCH/pictoSeq_results"
    echo "🕒 Completed at: $(date)"
    echo ""
    echo "🎯 Next steps:"
    echo "   1. Review test results"
    echo "   2. Submit full experiment with run_propicto_p100.slurm"
    echo "   3. Monitor job progress with: squeue --clusters=genius --user=\$(whoami)"
else
    echo ""
    echo "❌ ProPicto TEST failed!"
    echo "🕒 Failed at: $(date)"
    echo "🔍 Check logs for debugging"
    exit 1
fi

echo ""
echo "📋 Test Summary:"
echo "   Job ID: $SLURM_JOB_ID"
echo "   Node: $SLURMD_NODENAME" 
echo "   GPU: P100"
echo "   Duration: $(date)"
echo "   Results: $VSC_SCRATCH/pictoSeq_results"
echo ""
echo "🧪 Test finished!"