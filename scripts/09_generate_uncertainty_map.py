import h5py
import numpy as np
import SimpleITK as sitk

from pathlib import Path

from dicom_utils import get_shapes_from_dicom, find_respective_dicom_dir
from utils import setup_logger
from reconstruction_post_processing import norm_rescale01, post_process_3d_image


##################### PARAMETERS ############################
VSHARP_READER_STUDY_ROOT = Path('/scratch/hb-pca-rad/projects/03_reader_set_v2')
LOGDIR = Path('logs/')

# vSHARP Reconstruction Root Directories
ACC_ROOTS = {
    3: Path(f"/scratch/hb-pca-rad/projects/04_uncertainty_quantification/gaussian/recons_{3}x"),
    6: Path(f"/scratch/hb-pca-rad/projects/04_uncertainty_quantification/gaussian/recons_{6}x")
}

# Location where the .h5 kspace files are stored for each patient
KSPACE_ROOT_DIR = Path('/scratch/p290820/datasets/003_umcg_pst_ksps')      # source_dir

# Databases 
DB_FPATH_OLD = Path('/scratch/p290820/datasets/003_umcg_pst_ksps/database/dbs/master_habrok_20231106_v2.db')               # References an OLDER version of the databases where the info could also just be fine that we are looking for
DB_FPATH_NEW = Path('/home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/databases/master_habrok_20231106_v2.db') # References the LATEST version of the databases where the info could also just be fine that we are looking for

# Parameters
DEBUG              = True
DO_TESTS           = True
ACC_FACTORS        = [1, 3, 6] # Define the set of acceleration factors we care about.
DO_NORM            = True      # RESCALE 01
DO_POST_PROCESSING = True      # Post-processing, for each sub-reconstruction, such as k-space interpolation, flipping, and cropping
SAVE_AS_NIFTI_SUBR = True      # Save each sub-reconstruction as a NIFTI file
DECIMALS           = 4         # Number of decimals to round to

LOGGER = setup_logger(LOGDIR, use_time=False, part_fname='post_process_inference')

# All patient IDs to consider for Uncertainty Quantification
PAT_IDS = [
    '0003_ANON5046358',
    # '0004_ANON9616598',
    # '0005_ANON8290811',
    # '0006_ANON2379607',
    # '0007_ANON1586301',
    # '0008_ANON8890538',
    # '0010_ANON7748752',
    # '0011_ANON1102778',
    # '0012_ANON4982869',
    # '0013_ANON7362087',
    # '0014_ANON3951049',
    # '0015_ANON9844606',
    # '0018_ANON9843837',
    # '0019_ANON7657657',
    # '0020_ANON1562419',
    # '0021_ANON4277586',
    # '0023_ANON6964611',
    # '0024_ANON7992094',
    # '0026_ANON3620419',
    # '0027_ANON9724912',
    # '0028_ANON3394777',
    # '0029_ANON7189994',
    # '0030_ANON3397001',
    # '0031_ANON9141039',
    # '0032_ANON7649583',
    # '0033_ANON9728185',
    # '0035_ANON3474225',
    # '0036_ANON0282755',
    # '0037_ANON0369080',
    # '0039_ANON0604912',
    # '0042_ANON9423619',
    # '0043_ANON7041133',
    # '0044_ANON8232550',
    # '0045_ANON2563804',
    # '0047_ANON3613611',
    # '0048_ANON6365688',
    # '0049_ANON9783006',
    # '0051_ANON1327674',
    # '0052_ANON9710044',
    # '0053_ANON5517301',
    # '0055_ANON3357872',
    # '0056_ANON2124757',
    # '0057_ANON1070291',
    # '0058_ANON9719981',
    # '0059_ANON7955208',
    # '0061_ANON7642254',
    # '0062_ANON0319974',
    # '0063_ANON9972960',
    # '0064_ANON0282398',
    # '0067_ANON0913099',
    # '0068_ANON7978458',
    # '0069_ANON9840567',
    # '0070_ANON5223499',
    # '0071_ANON9806291',
    # '0073_ANON5954143',
    # '0075_ANON5895496',
    # '0076_ANON3983890',
    # '0077_ANON8634437',
    # '0078_ANON6883869',
    # '0079_ANON8828023',
    # '0080_ANON4499321',
    # '0081_ANON9763928',
    # '0082_ANON6073234',
    # '0083_ANON9898497',
    # '0084_ANON6141178',
    # '0085_ANON4535412',
    # '0086_ANON8511628',
    # '0087_ANON9534873',
    # '0088_ANON9892116',
    # '0089_ANON9786899',
    # '0090_ANON0891692',
    # '0092_ANON9941969',
    # '0093_ANON9728761',
    # '0094_ANON8024204',
    # '0095_ANON4189062',
    # '0097_ANON5642073',
    # '0103_ANON8583296',
    # '0104_ANON7748630',
    # '0105_ANON9883201',
    # '0107_ANON4035085',
    # '0108_ANON0424679',
    # '0109_ANON9816976',
    # '0110_ANON8266491',
    # '0111_ANON9310466',
    # '0112_ANON3210850',
    # '0113_ANON9665113',
    # '0115_ANON0400743',
    # '0116_ANON9223478',
    # '0118_ANON7141024',
    # '0119_ANON3865800',
    # '0120_ANON7275574',
    # '0121_ANON9629161',
    # '0123_ANON7265874',
    # '0124_ANON8610762',
    # '0125_ANON0272089',
    # '0126_ANON4747182',
    # '0127_ANON8023509',
    # '0128_ANON8627051',
    # '0129_ANON5344332',
    # '0135_ANON9879440',
    # '0136_ANON8096961',
    # '0137_ANON8035619',
    # '0138_ANON1747790',
    # '0139_ANON2666319',
    # '0140_ANON0899488',
    # '0141_ANON8018038',
    # '0142_ANON7090827',
    # '0143_ANON9752849',
    # '0144_ANON2255419',
    # '0145_ANON0335209',
    # '0146_ANON7414571',
    # '0148_ANON9604223',
    # '0149_ANON4712664',
    # '0150_ANON5824292',
    # '0152_ANON2411221',
    # '0153_ANON5958718',
    # '0155_ANON7828652',
    # '0157_ANON9873056',
    # '0159_ANON9720717',
    # '0160_ANON3504149'
]
#############################################################################






def load_patient_reconstructions(
        pat_root: Path  = None,
        norm            = False,
        do_post_process = True,
        acceleration    = None,
        save_nifti      = False,
        ref_nifti       = None,
        debug           = False
    ) -> np.ndarray:
    """
    Load and process the reconstructions for a patient directory containing .h5 files.
    it will load the reconstructions, normalize them to [0, 1], apply post-processing, and save them as Nifti files.
    There will be multiple reconstructions for each patient, depending on the number of .h5 files.
    
    Args:
        pat_root (Path): Path to the patient directory containing .h5 files.
        norm (bool): Normalize the reconstructions to [0, 1].
        do_post_process (bool): Apply post-processing to the reconstructions.
        acceleration (int): Acceleration factor (3 or 6) for post-processing.
        save_nifti (bool): Save the reconstructions as Nifti files.
        ref_nifti (np.ndarray): Reference Nifti copying of image infomation to make the reconstructions in the same spacing as the reference.
        debug (bool): Print debug information.
    
    Returns:
        np.ndarray: 4D NumPy array with shape (num_files, slices, rows, cols).
    """
    if DO_TESTS:
        assert pat_root.is_dir(), f"pat_root must be a directory, got: {pat_root}"
        assert isinstance(norm, bool), "norm must be a boolean."
        assert isinstance(do_post_process, bool), "do_post_process must be a boolean."
        assert acceleration in [3, 6], "acceleration must be None, 3, or 6."
        assert isinstance(debug, bool), "debug must be a boolean."
        assert isinstance(save_nifti, bool), "save_nifti must be a boolean."
        assert ref_nifti is None or isinstance(ref_nifti, sitk.Image), "ref_nifti must be None or a SimpleITK image."
        assert not save_nifti or ref_nifti is not None, "save_nifti and ref_nifti must be true toghether."

    h5_files = list(pat_root.glob('*.h5'))
    print(f"\nProcessing patient directory: {pat_root}, Found {len(h5_files)} .h5 files") if debug else None

    pat_id = pat_root.name
    dicom_dir, _ = find_respective_dicom_dir(pat_id, KSPACE_ROOT_DIR, DB_FPATH_OLD)
    zero_pad_shape, image_space_crop = get_shapes_from_dicom(dicom_dir)

    recons = []
    for idx, gaus_h5_file in enumerate(h5_files):
        print(f'\nProcessing: {gaus_h5_file.name}, with gaussian_id: {str(gaus_h5_file.stem).split("_")[-1]} ({idx+1}/{len(h5_files)})')
        
        with h5py.File(gaus_h5_file, 'r') as f:

            # 1) Loading
            recon_np = f['reconstruction'][()]
            # 2) Normalization
            if norm:
                recon_np = norm_rescale01(recon_np, debug=debug)
            # 3) Post-processing
            if do_post_process:
                recon_np = post_process_3d_image(recon_np, zero_pad_shape, image_space_crop)
            # 4) Save Nifti
            if save_nifti and ref_nifti is not None:
                recon_sitk = sitk.GetImageFromArray(recon_np)
                recon_sitk.CopyInformation(ref_nifti)
                dcml = "dcml" if do_post_process else ""
                recon_nifti_path = gaus_h5_file.parent / f"vsharp_r6_recon_{dcml}_{str(gaus_h5_file.stem).split('_')[-1]}.nii.gz"
                sitk.WriteImage(recon_sitk, str(recon_nifti_path))
                print(f"Saved reconstruction to: {recon_nifti_path}")
            if debug:
                print(f"Shape: {recon_np.shape}, Max: {recon_np.max():.7f}, Min: {recon_np.min():.7f}, Mean: {recon_np.mean():.7f}, Std: {recon_np.std():.7f}")
            recons.append(recon_np)

    recons_4d = np.stack(recons)
    print(f"Shape of recons_4d: {recons_4d.shape}") if debug else None
    return recons_4d






if __name__ == "__main__":
    print("Generating uncertainty map")

    ################################ Data Structures ################################
    # We'll store single reconstructions in one dictionary and stacks in another.
    # Key = acceleration factor, Value = list of 3D or 4D arrays (one per patient).
    single_recons_dict = {1: [], 3: [], 6: []}
    stack_recons_dict = {3: [], 6: []}  # R=1 doesn't have a stack of reconstructions.

    # Loop over each patient and each acceleration factor
    for pat_id in PAT_IDS:
        for acc in ACC_FACTORS:
            print(f"\nLoading reconstructions for patient: {pat_id}, Acceleration: {acc}")

            if acc == 1:
                # R=1: load the single reference reconstruction (e.g., _rss_target_dcml.mha)
                r1_fname = VSHARP_READER_STUDY_ROOT / pat_id / f"{pat_id}_rss_target_dcml.mha"
                recon_r1_sitk = sitk.ReadImage(str(r1_fname))
                recon_r1_arr = sitk.GetArrayFromImage(recon_r1_sitk)
                single_recons_dict[1].append(recon_r1_arr)
            
            else:
                # R=3 or R=6: load the single recon AND the stack of reconstructions
                recon_fname = VSHARP_READER_STUDY_ROOT / pat_id / f"{pat_id}_VSharp_R{acc}_recon_dcml.mha"
                recon_sitk = sitk.ReadImage(str(recon_fname))
                recon_arr = sitk.GetArrayFromImage(recon_sitk)
                single_recons_dict[acc].append(recon_arr)
                
                # Load the stack (multiple reconstructions for uncertainty quantification)
                # Use your load_patient_reconstructions function
                # For R=3 or R=6, we map them to ACC_ROOTS[acc].
                recon_stack_4d = load_patient_reconstructions(
                    pat_root        = ACC_ROOTS[acc] / pat_id,
                    norm            = DO_NORM,
                    do_post_process = DO_POST_PROCESSING,
                    acceleration    = acc,
                    save_nifti      = SAVE_AS_NIFTI_SUBR,
                    ref_nifti       = recon_sitk,   # optional reference SITK image
                    debug           = DEBUG
                )
                stack_recons_dict[acc].append(recon_stack_4d)

    print("\nDone loading reconstructions.")
    print("Single Recon Shapes:")
    for acc in single_recons_dict:
        print(f"  R={acc}: {len(single_recons_dict[acc])} items loaded.")

    print("Stack Recon Shapes (for R=3, R=6):")
    for acc in [3, 6]:
        print(f"  R={acc}: {len(stack_recons_dict[acc])} stacks loaded.")