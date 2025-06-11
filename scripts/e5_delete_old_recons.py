from pathlib import Path

if __name__ == "__main__":
    root_dir = Path('/scratch/hb-pca-rad/projects/04_uncertainty_quantification/gaussian/recons_3x')

    for subfolder in root_dir.iterdir():
        if subfolder.is_dir():
            for file in subfolder.glob('vsharp_r6_recon_dcml_gaus*.nii.gz'):
                try:
                    file.unlink()
                    print(f"Deleted: {file.resolve()}")
                except Exception as e:
                    print(f"Failed to delete {file.resolve()}: {e}")
