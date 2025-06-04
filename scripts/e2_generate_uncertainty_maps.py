import h5py
import numpy as np
import SimpleITK as sitk
import hashlib


from typing import List, Dict
from pathlib import Path

from assets.dicom_utils import get_shapes_from_dicom, find_respective_dicom_dir
from assets.utils import setup_logger
from assets.reconstruction_post_processing import norm_rescale01, post_process_3d_image
from assets.uncertainty_quantification import calculate_uncertainty_map

LOGGER = setup_logger(Path('/home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/logs/'), use_time=False, part_fname='post_process_inference')


def generate_uncertainty_map(
        pat_root: Path            = None,
        do_norm: bool             = False,
        do_post_process: bool     = True,
        do_round_sub_recons: bool = True,
        acceleration: int         = None,
        save_nifti: bool          = False,
        ref_nifti: sitk.Image     = None,
        kspace_root_dir: Path     = None,
        db_fpath_old: Path        = None,
        do_round_uq_map: bool     = False,
        decimals: int             = 4,
        uq_save_fpath: Path       = None, 
        debug: bool               = False,
    ) -> None:
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
    if True:
        assert pat_root.is_dir(), f"pat_root must be a directory, got: {pat_root}"
        assert isinstance(do_norm, bool), "norm must be a boolean."
        assert isinstance(do_post_process, bool), "do_post_process must be a boolean."
        assert acceleration in [3, 6], "acceleration must be None, 3, or 6."
        assert isinstance(debug, bool), "debug must be a boolean."
        assert isinstance(save_nifti, bool), "save_nifti must be a boolean."
        assert ref_nifti is None or isinstance(ref_nifti, sitk.Image), "ref_nifti must be None or a SimpleITK image."
        assert not save_nifti or ref_nifti is not None, "save_nifti and ref_nifti must be true toghether."
        assert uq_save_fpath is None or isinstance(uq_save_fpath, Path), "uq_save_fpath must be None or a Path."
        assert isinstance(do_round_uq_map, bool), "do_round_uq_map must be a boolean."
        assert isinstance(decimals, int), "decimals must be an integer."
        assert isinstance(kspace_root_dir, Path), "kspace_root_dir must be a Path."
        assert isinstance(db_fpath_old, Path), "db_fpath_old must be a Path."
        assert pat_root.is_dir(), f"pat_root must be a directory, got: {pat_root}"
        assert pat_root.exists(), f"pat_root does not exist: {pat_root}"
        assert pat_root.is_dir(), f"pat_root must be a directory, got: {pat_root}"

    h5_files = list(pat_root.glob('*.h5'))
    print(f"\n\tProcessing patient directory: {pat_root}, Found {len(h5_files)} .h5 files") if debug else None

    pat_id = pat_root.name
    dicom_dir, _ = find_respective_dicom_dir(pat_id, kspace_root_dir, db_fpath_old)
    zero_pad_shape, image_space_crop = get_shapes_from_dicom(dicom_dir)

    seen = set()
    recons = []
    for idx, h5_fname in enumerate(h5_files):
        print(f'\tProcessing: ({idx+1}/{len(h5_files)}) {h5_fname.name}, with fold_id: {str(h5_fname.stem).split("_")[-1]}')
        with h5py.File(h5_fname, 'r') as f:
            recon_np = f['reconstruction'][()]              # 1) Loading
            if do_norm:                                     # 2) Normalization
                recon_np = norm_rescale01(recon_np, debug=debug)
            if do_post_process:                             # 3) Post-processing
                recon_np = post_process_3d_image(recon_np, zero_pad_shape, image_space_crop)
            if do_round_sub_recons:                         # 3.1) Rounding
                recon_np = np.round(recon_np, decimals)
            print(f"\tShape: {recon_np.shape}, Max: {recon_np.max():.7f}, Min: {recon_np.min():.7f}, Mean: {recon_np.mean():.7f}, Std: {recon_np.std():.7f}")
        
        # Detecting duplicates with a hash of the raw bytes to detect duplicates
        h = hashlib.md5(recon_np.tobytes()).hexdigest()
        if h in seen:
            print(f"\tSkipping duplicate from {h5_fname.name} with hash {h[:10]}...{h[-10:]}")
            continue
        if save_nifti and ref_nifti is not None:        # 4) Save Nifti if not seen before
            recon_sitk = sitk.GetImageFromArray(recon_np)
            recon_sitk.CopyInformation(ref_nifti)
            dcml = "dcml" if do_post_process else ""
            recon_nifti_path = h5_fname.parent / f"vsharp_r{acceleration}_recon_{dcml}_{str(h5_fname.stem).split('_')[-1]}.nii.gz"
            sitk.WriteImage(recon_sitk, str(recon_nifti_path))
            print(f"Saved reconstruction to: {recon_nifti_path}")
        seen.add(h)
        recons.append(recon_np)
    
    # 5) Calculate uncertainty map and round
    uq_map = calculate_uncertainty_map(np.stack(recons), method='cv', debug=debug)
    if do_round_uq_map and do_norm:
        print(f'\tRounding uncertainty map to {decimals} decimals')
        uq_map = np.round(uq_map, decimals)
    print(f"\tHashes: {[h[:10]+'...' for h in seen]}")

    # 6) Save the uncertainty map as a Nifti file
    uq_map_sitk = sitk.GetImageFromArray(uq_map)
    if ref_nifti is not None:
        uq_map_sitk.CopyInformation(ref_nifti)
    sitk.WriteImage(uq_map_sitk, str(uq_save_fpath))
    print(f'Saved uncertainty map as {uq_save_fpath}')

    return None      # Goal is to save the uncertainty map as a Nifti file


def process_all_uncertainty_maps(
        uq_method: str                 = None,
        pat_ids: List[str]             = None,
        acc_factors: List[int]         = [3, 6],
        vsharp_reader_study_root: Path = None,
        acc_roots: Dict[int, Path]     = None,
        do_norm: bool                  = True,
        do_post_processing: bool       = True,
        do_round_sub_recons: bool      = True,
        save_subr_as_nifti: bool       = True,
        kspace_root_dir: Path          = None,
        db_fpath_old: Path             = None,
        do_round_uq_map: bool          = True,
        decimals: int                  = 4,
        debug: bool                    = False,
):
    """
    Process all patients to generate uncertainty maps from Gaussian reconstructions.
    
    For each patient and for each acceleration factor (3 or 6), it loads the reconstructions,
    applies normalization, post-processing, and generates the uncertainty map.
    Args:
        pat_ids (List[str]): List of patient IDs to process.
        acc_factors (List[int]): List of acceleration factors to consider.
        vsharp_reader_study_root (Path): Root directory for vSHARP reader study.
        acc_roots (Dict[int, Path]): Dictionary mapping acceleration factors to their respective directories.
        do_norm (bool): Normalize the reconstructions to [0, 1].
        do_post_processing (bool): Apply post-processing to the reconstructions.
        save_subr_as_nifti (bool): Save the sub-reconstructions as Nifti files.
        debug (bool): Print debug information.

    Returns:
        None
    """
    print(f"Processing all patients ({len(pat_ids)}) for UQ maps method=({uq_method})... for R(s): {acc_factors}")

    for pat_id in pat_ids:
        for acc in acc_factors:
            print(f"\nLoading reconstructions for patient: {pat_id}, Acceleration: {acc}, {uq_method}")
            
            # R=3 or R=6: load the reconstruction from the reader study
            actual_acc_recon = vsharp_reader_study_root / pat_id / f"{pat_id}_VSharp_R{acc}_recon_dcml.mha"
            
            # filename of the uq map to be made.
            target_uq_fpathname = acc_roots[acc] / pat_id / f"uq_map_R{acc}_{uq_method}.nii.gz"            # gm25. is ' gaussian method with 2.5 sigma' for the noise multiplier. but that must change

            # if this file already exists, we skip and continue
            if target_uq_fpathname.exists():
                print(f"Uncertainty map (R={acc}) already exists: {target_uq_fpathname}, skipping...")
                continue

            # Create the np.stack4d (multiple reconstructions for uncertainty quantification)
            generate_uncertainty_map(
                pat_root            = acc_roots[acc] / pat_id,
                do_norm             = do_norm,
                do_post_process     = do_post_processing,
                do_round_sub_recons = do_round_sub_recons,
                acceleration        = acc,
                save_nifti          = save_subr_as_nifti,
                ref_nifti           = sitk.ReadImage(str(actual_acc_recon)),   # optional reference SITK image
                kspace_root_dir     = kspace_root_dir,
                db_fpath_old        = db_fpath_old,
                do_round_uq_map     = do_round_uq_map,
                decimals            = decimals,
                uq_save_fpath       = target_uq_fpathname,     
                debug               = debug
            )


if __name__ == "__main__":
    print("GENERATING UNCERTAINTY MAPS FOR ALL PATIENTS")

    # All patient IDs to consider for Uncertainty Quantification
    pat_ids = [
        # '0003_ANON5046358',
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
        # '0027_ANON9724912',         # batch 1
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
        # '0079_ANON8828023',     # this one has been run. GAUSSIAN 
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
        '0108_ANON0424679',   #batch 5
        '0109_ANON9816976',
        '0110_ANON8266491',
        '0111_ANON9310466',
        '0112_ANON3210850',
        '0113_ANON9665113',
        '0115_ANON0400743',
        '0116_ANON9223478',
        # '0118_ANON7141024',       # has been removed from LXO generation due to an an issue with the central kspace line.
        '0119_ANON3865800',
        '0120_ANON7275574',
        '0121_ANON9629161',
        '0123_ANON7265874',
        '0124_ANON8610762',
        '0125_ANON0272089',
        '0126_ANON4747182',
        '0127_ANON8023509',
        '0128_ANON8627051',
        '0129_ANON5344332',
        '0135_ANON9879440',     #batch 5
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

    vsharp_reader_study_root = Path('/scratch/hb-pca-rad/projects/03_reader_set_v2')

    uq_methods = ['gaussian', 'lxo']  # 'gaussian' or # Leave-X-Out, where x are echo trains. for example L2O means that we leave out 2 echo trains.
    uq_methods = ['gaussian']              # 'gaussian' or # Leave-X-Out, where x are echo trains. for example L2O means that we leave out 2 echo trains.
    uq_methods = ['lxo']              # 'gaussian' or # Leave-X-Out, where x are echo trains. for example L2O means that we leave out 2 echo trains.
    
    # Parameters
    debug               = True
    do_round_sub_recons = True      # Round the sub-reconstructions to a certain number of decimals
    do_round_uq_map     = True      # Round the uncertainty map to a certain number of decimals
    acc_factors         = [3, 6] # Define the set of acceleration factors we care about.
    do_norm             = True      # Normalize the reconstructions to [0, 1]
    do_post_processing  = True      # Post-processing, for each sub-reconstruction, such as k-space interpolation, flipping, and cropping
    save_subr_as_nifti  = True      # Save each sub-reconstructions as NIFTI files
    decimals            = 4         # Number of decimals to round to
    
    # Databases 
    db_fpath_old = Path('/scratch/p290820/datasets/003_umcg_pst_ksps/database/dbs/master_habrok_20231106_v2.db')               # References an OLDER version of the databases where the info could also just be fine that we are looking for
    db_fpath_new = Path('/home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/databases/master_habrok_20231106_v2.db') # References the LATEST version of the databases where the info could also just be fine that we are looking for
    
    # Location where the .h5 kspace files are stored for each patient
    kspace_root_dir = Path('/scratch/p290820/datasets/003_umcg_pst_ksps')      # source_dir

    for uq_method in uq_methods:
        print(f"Processing uncertainty maps for method: {uq_method}")
        acc_roots = {        # vSHARP Reconstruction Root Directories
            3: Path(f"/scratch/hb-pca-rad/projects/04_uncertainty_quantification/{uq_method}/recons_{3}x"),
            6: Path(f"/scratch/hb-pca-rad/projects/04_uncertainty_quantification/{uq_method}/recons_{6}x")
        }

        process_all_uncertainty_maps(
            uq_method                = uq_method,
            pat_ids                  = pat_ids,
            acc_factors              = acc_factors,
            vsharp_reader_study_root = vsharp_reader_study_root,
            acc_roots                = acc_roots,
            do_norm                  = do_norm,
            do_post_processing       = do_post_processing,
            do_round_sub_recons      = do_round_sub_recons,
            save_subr_as_nifti       = save_subr_as_nifti,
            kspace_root_dir          = kspace_root_dir,
            db_fpath_old             = db_fpath_old,
            do_round_uq_map          = do_round_uq_map,
            decimals                 = decimals,
            debug                    = debug
        )