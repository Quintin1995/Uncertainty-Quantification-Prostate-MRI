#!/bin/bash -l
#SBATCH --job-name=uq_vsharp_lxo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=3,6              # R values: 3× and 6× acceleration
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=q.y.van.lohuizen@umcg.nl
#SBATCH --output=/scratch/hb-pca-rad/projects/04_uncertainty_quantification/logs/uq_vsharp_R%a_%A.out
#SBATCH --error=/scratch/hb-pca-rad/projects/04_uncertainty_quantification/logs/uq_vsharp_R%a_%A.err

set -euo pipefail
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load CUDA/11.7.0
module load Python/3.9.6-GCCcore-11.2.0
module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0
source /home1/p290820/repos/direct-with-averages/env/bin/activate

# --------------------------------------------------
# Configuration parameters
# --------------------------------------------------
# Number of leave-one-out folds you want per R
NUM_FOLDS=6

# Paths
DATA="/scratch/p290820/datasets/003_umcg_pst_ksps/data"
CKPT="/scratch/hb-pca-rad/projects/03_nki_reader_study/checkpoints/model_152000.pt"
LIST="/home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/lists/split_by_20/umcg_0001_0172_1.lst"

# --------------------------------------------------
# Derive variables
# --------------------------------------------------
R=$SLURM_ARRAY_TASK_ID
PRED_DIR="/scratch/hb-pca-rad/projects/04_uncertainty_quantification/lxo/recons_R${R}x"
mkdir -p "${PRED_DIR}"

echo "[$(date +'%F %T')] Starting UQ recon: R=${R}× with ${NUM_FOLDS} folds"

# --------------------------------------------------
# Run reconstructions for each fold
# --------------------------------------------------
for (( FOLD=0; FOLD<NUM_FOLDS; FOLD++ )); do
  echo "[$(date +'%T')] R=${R} fold=${FOLD}"
  srun direct predict "${PRED_DIR}" \
    --cfg "/home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/configs/vsharp/vsharp_r${R}_lxo_fold${FOLD}.yaml" \
    --data-root "${DATA}" \
    --checkpoint "${CKPT}" \
    --filenames-filter "${LIST}" \
    --num-gpus 1 \
    --num-workers $SLURM_CPUS_PER_TASK
done

echo "[$(date +'%F %T')] Completed UQ recon: R=${R}×"
