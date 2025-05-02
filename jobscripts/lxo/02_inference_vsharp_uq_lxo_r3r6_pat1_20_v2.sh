#!/bin/bash -l
#SBATCH --job-name=uq_vsharp_R3-6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=12:00:00              # allow time for both sets
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=q.y.van.lohuizen@umcg.nl
#SBATCH --output=/scratch/hb-pca-rad/projects/04_uncertainty_quantification/logs/uq_vsharp_R%j.out
#SBATCH --error=/scratch/hb-pca-rad/projects/04_uncertainty_quantification/logs/uq_vsharp_R%j.err

set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load CUDA/11.7.0 Python/3.9.6-GCCcore-11.2.0 NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0
source /home1/p290820/repos/direct-with-averages/env/bin/activate

cd /home1/p290820/repos/direct-with-averages

# common paths
DATA=/scratch/p290820/datasets/003_umcg_pst_ksps/data
CKPT=/scratch/hb-pca-rad/projects/03_nki_reader_study/checkpoints/model_152000.pt
LIST=/home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/lists/split_by_20/umcg_0001_0172_1.lst

# Function to run N× reconstruction
run_recon() {
  local R=$1          # down‐sampling factor (3 or 6)
  local PRED_DIR="/scratch/hb-pca-rad/projects/04_uncertainty_quantification/lxo/recons_${R}x"
  local PREFIX="vsharp_r${R}_lxo_fold"

  for FOLD in {0..6}; do
    echo "[$(date +'%H:%M:%S')] Starting R=${R} fold=${FOLD}"
    srun direct predict "${PRED_DIR}" \
      --cfg "/home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/configs/vsharp/${PREFIX}${FOLD}.yaml" \
      --data-root "${DATA}" \
      --checkpoint "${CKPT}" \
      --filenames-filter "${LIST}" \
      --num-gpus 1 \
      --num-workers $SLURM_CPUS_PER_TASK
    echo "[$(date +'%H:%M:%S')] Finished R=${R} fold=${FOLD}"
  done
}

# Run R=3, then R=6
run_recon 3
run_recon 6
