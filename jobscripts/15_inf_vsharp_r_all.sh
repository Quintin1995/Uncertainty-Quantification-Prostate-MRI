#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=96000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=q.y.van.lohuizen@umcg.nl
#SBATCH --job-name=debugR2_R8
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/hb-pca-rad/projects/04_personalized_recon/logs/job-%j_inf_vshp_r_all_debug.log

# Go to the relevant project directory
# cd /home1/p290820/repos/direct
cd /home1/p290820/repos/direct-with-averages

# Remove all currently loaded modules from the module environment
module purge
module load CUDA/11.7.0
module load Python/3.9.6-GCCcore-11.2.0
module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0

source /home1/p290820/repos/direct-with-averages/env/bin/activate



# NOTE: It seems that the interactive GPU 2 cannot run the program with multi treading, but requesting another GPU works fine.

# PART DEBUG TEST
direct predict /scratch/hb-pca-rad/projects/04_personalized_recon/reconstructions/debug \
    --cfg /scratch/hb-pca-rad/projects/04_personalized_recon/configs/vsharp_r_all_debug.yaml \
    --data-root /scratch/p290820/datasets/003_umcg_pst_ksps/data \
    --checkpoint /scratch/hb-pca-rad/projects/03_nki_reader_study/checkpoints/model_152000.pt \
    --filenames-filter /scratch/hb-pca-rad/projects/04_personalized_recon/lists/umcg_0001_0172_1_debug.lst \
    --num-gpus 1 \
    --num-workers 8


# PART 1 - 15 patients
direct predict /scratch/hb-pca-rad/projects/04_personalized_recon/reconstructions/debug \
    --cfg /scratch/hb-pca-rad/projects/04_personalized_recon/configs/vsharp_r_all_debug.yaml \
    --data-root /scratch/p290820/datasets/003_umcg_pst_ksps/data \
    --checkpoint /scratch/hb-pca-rad/projects/03_nki_reader_study/checkpoints/model_152000.pt \
    --filenames-filter /scratch/hb-pca-rad/projects/04_personalized_recon/lists/split_by_15/umcg_0001_0172_1.lst \
    --num-gpus 1 \
    --num-workers 1

