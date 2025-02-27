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
