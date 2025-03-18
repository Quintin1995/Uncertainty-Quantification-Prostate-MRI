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
2. Get an interactive v100 with: 
`alias resa100_30="srun --gres=gpu:a100:1 --exclude=a100gpu2 --partition=gpu --time=00:30:00 --cpus-per-task=8 --mem=64000 --pty /bin/bash"
alias resa100_30="srun --gres=gpu:a100:1 --partition=gpu --time=00:30:00 --cpus-per-task=8 --mem=64000 --pty /bin/bash"
alias resa100_60="srun --gres=gpu --exclude=a100gpu3 --partition=gpu --time=01:00:00 --cpus-per-task=12 --mem=64000 --pty /bin/bash"
alias resa100_90="srun --gres=gpu --exclude=a100gpu3 --partition=gpu --time=01:30:00 --cpus-per-task=12 --mem=64000 --pty /bin/bash"
alias resa100_120="srun --gres=gpu --exclude=a100gpu3 --partition=gpu --time=02:00:00 --cpus-per-task=12 --mem=64000 --pty /bin/bash"

alias resv100_60="srun --gres=gpu:v100:1 --partition=gpu --time=01:00:00 --cpus-per-task=8 --mem=64000 --pty /bin/bash"
alias resv100_30="srun --gres=gpu:v100:1 --partition=gpu --time=00:30:00 --cpus-per-task=8 --mem=32000 --pty /bin/bash"`

