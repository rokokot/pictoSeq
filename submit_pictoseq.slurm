#!/bin/bash -l
#SBATCH --clusters=genius
#SBATCH --account=intro_vsc37132
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu_p100
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH --job-name=pictoseq_p100
#SBATCH --output=pictoseq_logs/pictoseq_p100_main_%j.out
#SBATCH --error=pictoseq_logs/pictoseq_p100_main_%j.err

mkdir -p pictoseq_logs
echo "PictoSeq Research Pipeline - Job $SLURM_JOB_ID"
echo "Started: $(date)"

module purge
module load miniconda3
source activate pictoseq

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

cd $VSC_DATA/pictoSeq

if [ ! -d "data/processed_propicto" ]; then
    echo "Error: data/processed_propicto not found"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "Starting experimental matrix: 3 models x 4 data configs = 12 experiments"

python main_runner.py \
    --run-all \
    --results-path "$VSC_SCRATCH/pictoseq_results" \
    --max-train 10000 \
    --max-val 1000 \
    --max-test 1000 \
    --epochs 5 \
    --batch-size 8 \
    --learning-rate 3e-5

if [ $? -eq 0 ]; then
    echo "Completed successfully: $(date)"
    echo "Results: $VSC_SCRATCH/pictoseq_results"
else
    echo "Failed: $(date)"
    exit 1
fi