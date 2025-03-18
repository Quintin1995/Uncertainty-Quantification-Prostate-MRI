#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=96000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=q.y.van.lohuizen@umcg.nl
#SBATCH --job-name=InfR6vSHARP
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/hb-pca-rad/qvl/logs/job-%j_inf_vshp_r6_1avgr2_v1.log

# Go to the relevant project directory
cd /home1/p290820/repos/direct

# Remove all currently loaded modules from the module environment
module purge
module load CUDA/11.7.0
module load Python/3.9.6-GCCcore-11.2.0
module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0

source /home1/p290820/repos/direct/env/bin/activate


# PART 1 - 15 patients
direct predict /scratch/p290820/projects/03_nki_reader_study/output/umcg/6x \
    --cfg /scratch/p290820/projects/03_nki_reader_study/configs/vsharp_r6.yaml \
    --data-root /scratch/p290820/datasets/003_umcg_pst_ksps/data \
    --checkpoint /scratch/p290820/projects/03_nki_reader_study/checkpoints/model_152000.pt \
    --filenames-filter /scratch/p290820/projects/03_nki_reader_study/lists/split_by_15/umcg_0001_0172_1.lst \
    --num-gpus 1 \
    --num-workers 6

# PART 2 - 15 patients
direct predict /scratch/p290820/projects/03_nki_reader_study/output/umcg/6x \
    --cfg /scratch/p290820/projects/03_nki_reader_study/configs/vsharp_r6.yaml \
    --data-root /scratch/p290820/datasets/003_umcg_pst_ksps/data \
    --checkpoint /scratch/p290820/projects/03_nki_reader_study/checkpoints/model_152000.pt \
    --filenames-filter /scratch/p290820/projects/03_nki_reader_study/lists/split_by_15/umcg_0001_0172_2.lst \
    --num-gpus 1 \
    --num-workers 6

# PART 3 - 15 patients
direct predict /scratch/p290820/projects/03_nki_reader_study/output/umcg/6x \
    --cfg /scratch/p290820/projects/03_nki_reader_study/configs/vsharp_r6.yaml \
    --data-root /scratch/p290820/datasets/003_umcg_pst_ksps/data \
    --checkpoint /scratch/p290820/projects/03_nki_reader_study/checkpoints/model_152000.pt \
    --filenames-filter /scratch/p290820/projects/03_nki_reader_study/lists/split_by_15/umcg_0001_0172_3.lst \
    --num-gpus 1 \
    --num-workers 6

# PART 4 - 15 patients
direct predict /scratch/p290820/projects/03_nki_reader_study/output/umcg/6x \
    --cfg /scratch/p290820/projects/03_nki_reader_study/configs/vsharp_r6.yaml \
    --data-root /scratch/p290820/datasets/003_umcg_pst_ksps/data \
    --checkpoint /scratch/p290820/projects/03_nki_reader_study/checkpoints/model_152000.pt \
    --filenames-filter /scratch/p290820/projects/03_nki_reader_study/lists/split_by_15/umcg_0001_0172_4.lst \
    --num-gpus 1 \
    --num-workers 6

# PART 5 - 15 patients
direct predict /scratch/p290820/projects/03_nki_reader_study/output/umcg/6x \
    --cfg /scratch/p290820/projects/03_nki_reader_study/configs/vsharp_r6.yaml \
    --data-root /scratch/p290820/datasets/003_umcg_pst_ksps/data \
    --checkpoint /scratch/p290820/projects/03_nki_reader_study/checkpoints/model_152000.pt \
    --filenames-filter /scratch/p290820/projects/03_nki_reader_study/lists/split_by_15/umcg_0001_0172_5.lst \
    --num-gpus 1 \
    --num-workers 6

# PART 6 - 15 patients
direct predict /scratch/p290820/projects/03_nki_reader_study/output/umcg/6x \
    --cfg /scratch/p290820/projects/03_nki_reader_study/configs/vsharp_r6.yaml \
    --data-root /scratch/p290820/datasets/003_umcg_pst_ksps/data \
    --checkpoint /scratch/p290820/projects/03_nki_reader_study/checkpoints/model_152000.pt \
    --filenames-filter /scratch/p290820/projects/03_nki_reader_study/lists/split_by_15/umcg_0001_0172_6.lst \
    --num-gpus 1 \
    --num-workers 6

# PART 7 - 15 patients
direct predict /scratch/p290820/projects/03_nki_reader_study/output/umcg/6x \
    --cfg /scratch/p290820/projects/03_nki_reader_study/configs/vsharp_r6.yaml \
    --data-root /scratch/p290820/datasets/003_umcg_pst_ksps/data \
    --checkpoint /scratch/p290820/projects/03_nki_reader_study/checkpoints/model_152000.pt \
    --filenames-filter /scratch/p290820/projects/03_nki_reader_study/lists/split_by_15/umcg_0001_0172_7.lst \
    --num-gpus 1 \
    --num-workers 6

# PART 8 - 15 patients
direct predict /scratch/p290820/projects/03_nki_reader_study/output/umcg/6x \
    --cfg /scratch/p290820/projects/03_nki_reader_study/configs/vsharp_r6.yaml \
    --data-root /scratch/p290820/datasets/003_umcg_pst_ksps/data \
    --checkpoint /scratch/p290820/projects/03_nki_reader_study/checkpoints/model_152000.pt \
    --filenames-filter /scratch/p290820/projects/03_nki_reader_study/lists/split_by_15/umcg_0001_0172_8.lst \
    --num-gpus 1 \
    --num-workers 6

# PART 9 - 15 patients
direct predict /scratch/p290820/projects/03_nki_reader_study/output/umcg/6x \
    --cfg /scratch/p290820/projects/03_nki_reader_study/configs/vsharp_r6.yaml \
    --data-root /scratch/p290820/datasets/003_umcg_pst_ksps/data \
    --checkpoint /scratch/p290820/projects/03_nki_reader_study/checkpoints/model_152000.pt \
    --filenames-filter /scratch/p290820/projects/03_nki_reader_study/lists/split_by_15/umcg_0001_0172_9.lst \
    --num-gpus 1 \
    --num-workers 6
