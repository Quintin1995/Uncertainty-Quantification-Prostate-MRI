# Uncertainty Quantification for Prostate MRI

This project performs uncertainty quantification for prostate MRI. Our goals are to:
- **Adaptively Reconstruct:** Adjust reconstruction based on temporal uncertainty.
- **Correlate Maps:** Compare error maps with uncertainty maps to improve diagnostic confidence.

---

## Project Components

### Code Repositories
- **Uncertainty-Quantification-Prostate-MRI**  
  Contains code for processing vSHARP Reconstructions.  
  *Location:* `/home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI`

- **DIRECT with Averages**  
  Hosts the DIRECT repository for creating the reconstructions.  
  *Location:* `/home1/p290820/repos/direct-with-averages`

### Reconstructions Storage
- **Reconstructions Directory**  
  Stores the generated reconstructions.  
  *Location:* `/scratch/hb-pca-rad/projects/04_personalized_recon`  
  > **Note:** The `/home1` partition has limited storage.

---

## Workspace Setup: Habrok SSH Connect

1. **SSH into GPU2:**  
   Log into the interactive GPU2 system via SSH.

2. **VSCode Remote SSH:**  
   Connect to GPU2 using VSCode's remote SSH extension.
   - Open the **Uncertainty-Quantification-Prostate-MRI** repo for processing vSHARP Reconstructions.
   - Open the **DIRECT with Averages** repo to generate reconstructions.
   - Access the reconstructions in the `/scratch/hb-pca-rad/projects/04_personalized_recon` directory.

---

## Additional Information

- **Storage Management:**  
  Due to limited space on `/home1`, reconstructions are stored on `/scratch`.

- **Further Documentation:**  
  Please refer to the individual repositories for more detailed instructions on usage and configuration.

- **Scripts:**
   The post-processing script(s) were taken from the Reader Study Project within the Repository: direct-with-averages


## Running

1. SSH into Habrok

2. Get an interactive GPU with:

### a100
```bash
srun --gres=gpu:a100:1 --exclude=a100gpu2 --partition=gpu --time=00:30:00 --cpus-per-task=8 --mem=64000 --pty /bin/bash
srun --gres=gpu:a100:1 --partition=gpu --time=00:30:00 --cpus-per-task=8 --mem=64000 --pty /bin/bash
srun --gres=gpu --exclude=a100gpu3 --partition=gpu --time=01:00:00 --cpus-per-task=12 --mem=64000 --pty /bin/bash
srun --gres=gpu --exclude=a100gpu3 --partition=gpu --time=01:30:00 --cpus-per-task=12 --mem=64000 --pty /bin/bash
srun --gres=gpu --exclude=a100gpu3 --partition=gpu --time=02:00:00 --cpus-per-task=12 --mem=64000 --pty /bin/bash
```

### v100
```bash
srun --gres=gpu:v100:1 --partition=gpu --time=01:00:00 --cpus-per-task=8 --mem=64000 --pty /bin/bash
srun --gres=gpu:v100:1 --partition=gpu --time=00:30:00 --cpus-per-task=8 --mem=32000 --pty /bin/bash
```

3. Run the environment and vSHARP reconstruction code:
```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=96000
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=q.y.van.lohuizen@umcg.nl
#SBATCH --job-name=uq_gaus
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/hb-pca-rad/projects/04_uncertainty_quantification/logs/job-%j_inf_vshp_gauss.log

# Go to the relevant project directory
# cd /home1/p290820/repos/direct
cd /home1/p290820/repos/direct-with-averages

# Remove all currently loaded modules from the module environment
module purge
module load CUDA/11.7.0
module load Python/3.9.6-GCCcore-11.2.0
module load NCCL/2.12.12-GCCcore-11.3.0-CUDA-11.7.0

# Environment loading
source /home1/p290820/repos/direct-with-averages/env/bin/activate

# NOTE: It seems that the interactive GPU 2 cannot run the program with multi treading, but requesting another GPU works fine.

# PART DEBUG TEST
direct predict /scratch/hb-pca-rad/projects/04_uncertainty_quantification/recons_gaussian_noise/debug \
    --cfg /home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/configs/vsharp/vsharp_r3_remade.yaml \
    --data-root /scratch/p290820/datasets/003_umcg_pst_ksps/data \
    --checkpoint /scratch/hb-pca-rad/projects/03_nki_reader_study/checkpoints/model_152000.pt \
    --filenames-filter /scratch/hb-pca-rad/projects/04_uncertainty_quantification/lists/umcg_0001_0172_1_debug.lst \
    --num-gpus 1 \
    --num-workers 8

# # PART 1 - 15 patients
# direct predict /scratch/hb-pca-rad/projects/04_uncertainty_quantification/reconstructions/debug \
#     --cfg /scratch/hb-pca-rad/projects/04_uncertainty_quantification/configs/vsharp_r_all_debug.yaml \
#     --data-root /scratch/p290820/datasets/003_umcg_pst_ksps/data \
#     --checkpoint /scratch/hb-pca-rad/projects/03_nki_reader_study/checkpoints/model_152000.pt \
#     --filenames-filter /scratch/hb-pca-rad/projects/04_uncertainty_quantification/lists/split_by_15/umcg_0001_0172_1.lst \
#     --num-gpus 1 \
#     --num-workers 1
```