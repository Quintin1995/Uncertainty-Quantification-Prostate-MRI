#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=96000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=q.y.van.lohuizen@umcg.nl
#SBATCH --job-name=uq_gaus_r3
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/hb-pca-rad/projects/04_uncertainty_quantification/logs/job-%j_inf_vshp_gauss_r3.log

# Go to the relevant project directory
# cd /home1/p290820/repos/direct
cd /home1/p290820/repos/direct-with-averages

# Remove all currently loaded modules from the module environment
module purge
module load CUDA/11.7.0
module load Python/3.9.6-GCCcore-11.2.0
module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0

source /home1/p290820/repos/direct-with-averages/env/bin/activate

# R3
direct predict /scratch/hb-pca-rad/projects/04_uncertainty_quantification/lxo/recons_6x \
    --cfg /home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/configs/vsharp/vsharp_r6_lxo_fold0.yaml \
    --data-root /scratch/p290820/datasets/003_umcg_pst_ksps/data \
    --checkpoint /scratch/hb-pca-rad/projects/03_nki_reader_study/checkpoints/model_152000.pt \
    --filenames-filter /home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/lists/split_by_20/umcg_0001_0172_1.lst \
    --num-gpus 1 \
    --num-workers 8

direct predict /scratch/hb-pca-rad/projects/04_uncertainty_quantification/lxo/recons_6x \
    --cfg /home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/configs/vsharp/vsharp_r6_lxo_fold1.yaml \
    --data-root /scratch/p290820/datasets/003_umcg_pst_ksps/data \
    --checkpoint /scratch/hb-pca-rad/projects/03_nki_reader_study/checkpoints/model_152000.pt \
    --filenames-filter /home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/lists/split_by_20/umcg_0001_0172_1.lst \
    --num-gpus 1 \
    --num-workers 8

direct predict /scratch/hb-pca-rad/projects/04_uncertainty_quantification/lxo/recons_6x \
    --cfg /home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/configs/vsharp/vsharp_r6_lxo_fold2.yaml \
    --data-root /scratch/p290820/datasets/003_umcg_pst_ksps/data \
    --checkpoint /scratch/hb-pca-rad/projects/03_nki_reader_study/checkpoints/model_152000.pt \
    --filenames-filter /home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/lists/split_by_20/umcg_0001_0172_1.lst \
    --num-gpus 1 \
    --num-workers 8

direct predict /scratch/hb-pca-rad/projects/04_uncertainty_quantification/lxo/recons_6x \
    --cfg /home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/configs/vsharp/vsharp_r6_lxo_fold3.yaml \
    --data-root /scratch/p290820/datasets/003_umcg_pst_ksps/data \
    --checkpoint /scratch/hb-pca-rad/projects/03_nki_reader_study/checkpoints/model_152000.pt \
    --filenames-filter /home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/lists/split_by_20/umcg_0001_0172_1.lst \
    --num-gpus 1 \
    --num-workers 8

direct predict /scratch/hb-pca-rad/projects/04_uncertainty_quantification/lxo/recons_6x \
    --cfg /home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/configs/vsharp/vsharp_r6_lxo_fold4.yaml \
    --data-root /scratch/p290820/datasets/003_umcg_pst_ksps/data \
    --checkpoint /scratch/hb-pca-rad/projects/03_nki_reader_study/checkpoints/model_152000.pt \
    --filenames-filter /home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/lists/split_by_20/umcg_0001_0172_1.lst \
    --num-gpus 1 \
    --num-workers 8

direct predict /scratch/hb-pca-rad/projects/04_uncertainty_quantification/lxo/recons_6x \
    --cfg /home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/configs/vsharp/vsharp_r6_lxo_fold5.yaml \
    --data-root /scratch/p290820/datasets/003_umcg_pst_ksps/data \
    --checkpoint /scratch/hb-pca-rad/projects/03_nki_reader_study/checkpoints/model_152000.pt \
    --filenames-filter /home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/lists/split_by_20/umcg_0001_0172_1.lst \
    --num-gpus 1 \
    --num-workers 8

direct predict /scratch/hb-pca-rad/projects/04_uncertainty_quantification/lxo/recons_6x \
    --cfg /home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/configs/vsharp/vsharp_r6_lxo_fold6.yaml \
    --data-root /scratch/p290820/datasets/003_umcg_pst_ksps/data \
    --checkpoint /scratch/hb-pca-rad/projects/03_nki_reader_study/checkpoints/model_152000.pt \
    --filenames-filter /home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/lists/split_by_20/umcg_0001_0172_1.lst \
    --num-gpus 1 \
    --num-workers 8

