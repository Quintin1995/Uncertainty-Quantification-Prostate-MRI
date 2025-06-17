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

LOGGER = setup_logger(Path('/home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/logs/'), use_time=False, part_fname='generate_uncertainty_maps')


def generate_uncertainty_map(
    pat_root: Path,
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
    uq_metric: str            = 'std',
) -> None:
    """
    Load sub-reconstructions, optionally normalize/post-process them, compute uncertainty map,
    and save as NIfTI using ref_nifti spacing.
    """

    # === Input validation ===
    if True:
        assert pat_root.is_dir(), f"❌ pat_root is not a directory: {pat_root}"
        assert acceleration in [3, 6], f"❌ Invalid acceleration: {acceleration}"
        assert not save_nifti or ref_nifti is not None, "❌ Need ref_nifti to save NIfTI"
    
    h5_files = list(pat_root.glob("*.h5"))
    LOGGER.info(f"\n📁 Processing: {pat_root.name} | {len(h5_files)} .h5 files found")

    pat_id = pat_root.name
    dicom_dir, _ = find_respective_dicom_dir(pat_id, kspace_root_dir, db_fpath_old)
    zero_pad_shape, image_space_crop = get_shapes_from_dicom(dicom_dir)

    seen = set()
    recons = []

    # === Load and process reconstructions ===
    for idx, h5_fname in enumerate(h5_files):
        fold_id = str(h5_fname.stem).split("_")[-1]
        LOGGER.info(f"🔄  ({idx+1}/{len(h5_files)}) Processing fold {fold_id}: {h5_fname.name}")
        
        with h5py.File(h5_fname, 'r') as f:
            recon_np = f['reconstruction'][()]
        
        if do_norm:
            recon_np = norm_rescale01(recon_np, debug=debug)
        if do_post_process:
            recon_np = post_process_3d_image(recon_np, zero_pad_shape, image_space_crop)
        if do_round_sub_recons:
            recon_np = np.round(recon_np, decimals)

        LOGGER.info(f"\t📐 Shape: {recon_np.shape} | μ={recon_np.mean():.6f}, σ={recon_np.std():.6f}, min={recon_np.min():.6f}, max={recon_np.max():.6f}")
        
        # Deduplication by hash
        h = hashlib.md5(recon_np.tobytes()).hexdigest()
        if h in seen:
            LOGGER.warning(f"\t⚠️  Duplicate found: {h5_fname.name} [hash {h[:10]}...] — Skipping")
            continue

        # Save intermediate recon if needed
        if save_nifti and ref_nifti:
            recon_sitk = sitk.GetImageFromArray(recon_np)
            recon_sitk.CopyInformation(ref_nifti)
            dcml = "dcml" if do_post_process else ""
            nifti_path = h5_fname.parent / f"vsharp_r{acceleration}_recon_{dcml}_{fold_id}.nii.gz"
            sitk.WriteImage(recon_sitk, str(nifti_path))
            LOGGER.info(f"\t💾 Saved recon NIfTI to: {nifti_path}")

        seen.add(h)
        recons.append(recon_np)

    # === Compute UQ map ===
    uq_map = calculate_uncertainty_map(np.stack(recons), method=uq_metric, debug=debug)
    if do_round_uq_map and do_norm:
        LOGGER.info(f"\t🧮 Rounding UQ map to {decimals} decimals")
        uq_map = np.round(uq_map, decimals)

    LOGGER.info(f"\t🔐 Unique hashes: {[h[:10]+'...' for h in seen]}")

    # === Save final UQ map ===
    uq_map_sitk = sitk.GetImageFromArray(uq_map)
    if ref_nifti:
        uq_map_sitk.CopyInformation(ref_nifti)
    sitk.WriteImage(uq_map_sitk, str(uq_save_fpath))
    LOGGER.info(f"✅ Saved final UQ map to: {uq_save_fpath}")


def process_all_uncertainty_maps(
    uq_method: str             = None,
    pat_ids: List[str]         = None,
    acc_factors: List[int]     = [3, 6],
    reader_study_root: Path    = None,
    acc_roots: Dict[int, Path] = None,
    do_norm: bool              = True,
    do_post_processing: bool   = True,
    do_round_sub_recons: bool  = True,
    save_subr_as_nifti: bool   = True,
    kspace_root_dir: Path      = None,
    db_fpath_old: Path         = None,
    do_round_uq_map: bool      = True,
    decimals: int              = 4,
    debug: bool                = False,
    uq_metric: str             = 'std',
):
    """
    Generate and save uncertainty maps per patient and acceleration factor.
    """

    LOGGER.info(f"🧪 Starting UQ Map Generation | Method: '{uq_method}' | Metric: '{uq_metric}' | Total Patients: {len(pat_ids)}")
    LOGGER.info(f"⚙️  Options: norm={do_norm}, postproc={do_post_processing}, round={do_round_sub_recons}, save_nifti={save_subr_as_nifti}")

    for pat_idx, pat_id in enumerate(pat_ids):
        LOGGER.info(f"\n🔍 [{pat_idx + 1}/{len(pat_ids)}] Processing patient: {pat_id}")

        for acc in acc_factors:
            LOGGER.info(f"  📉  Acceleration: R={acc}x | Method: {uq_method.upper()}")

            # Source recon path
            recon_path = reader_study_root / pat_id / f"{pat_id}_VSharp_R{acc}_recon_dcml.mha"

            # Target path for saving the uncertainty map
            uq_save_path = acc_roots[acc] / pat_id / f"uq_map_R{acc}_{uq_method}_{uq_metric}.nii.gz"

            if uq_save_path.exists():
                LOGGER.info(f"  ⏭️  UQ map already exists → Skipping: {uq_save_path}")
                continue

            LOGGER.info(f"  🛠️  Generating new UQ map → {uq_save_path}")

            generate_uncertainty_map(
                pat_root            = acc_roots[acc] / pat_id,
                do_norm             = do_norm,
                do_post_process     = do_post_processing,
                do_round_sub_recons = do_round_sub_recons,
                acceleration        = acc,
                save_nifti          = save_subr_as_nifti,
                ref_nifti           = sitk.ReadImage(str(recon_path)),
                kspace_root_dir     = kspace_root_dir,
                db_fpath_old        = db_fpath_old,
                do_round_uq_map     = do_round_uq_map,
                decimals            = decimals,
                uq_save_fpath       = uq_save_path,
                debug               = debug,
                uq_metric           = uq_metric,
            )

            LOGGER.info(f"  ✅ UQ map saved for {pat_id} @ R={acc}x\n")

    LOGGER.info(f"\n🏁 Finished processing all patients for UQ method '{uq_method}'\n{'='*60}")




# ------------------------------------------------------------------------------
# Main execution block
if __name__ == "__main__":

    # === Statistical Parameters for Uncertainty Quantification ===
    pat_ids     = [
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
        '0028_ANON3394777',
        '0029_ANON7189994',
        '0030_ANON3397001',
        '0031_ANON9141039',
        '0032_ANON7649583',
        '0033_ANON9728185',
        '0035_ANON3474225',
        '0036_ANON0282755',
        '0037_ANON0369080',
        '0039_ANON0604912',
        '0042_ANON9423619',
        '0043_ANON7041133',
        '0044_ANON8232550',
        '0045_ANON2563804',
        '0047_ANON3613611',
        '0048_ANON6365688',
        '0049_ANON9783006',
        '0051_ANON1327674',
        '0052_ANON9710044',
        '0053_ANON5517301',
        '0055_ANON3357872',
        '0056_ANON2124757',
        '0057_ANON1070291',
        '0058_ANON9719981',
        '0059_ANON7955208',
        '0061_ANON7642254',
        '0062_ANON0319974',
        '0063_ANON9972960',
        '0064_ANON0282398',
        '0067_ANON0913099',
        '0068_ANON7978458',
        '0069_ANON9840567',
        '0070_ANON5223499',
        '0071_ANON9806291',
        '0073_ANON5954143',
        '0075_ANON5895496',
        '0076_ANON3983890',
        '0077_ANON8634437',
        '0078_ANON6883869',
        '0079_ANON8828023',     # this one has been run. GAUSSIAN 
        '0080_ANON4499321',
        '0081_ANON9763928',
        '0082_ANON6073234',
        '0083_ANON9898497',
        '0084_ANON6141178',
        '0085_ANON4535412',
        '0086_ANON8511628',
        '0087_ANON9534873',
        '0088_ANON9892116',
        '0089_ANON9786899',
        '0090_ANON0891692',
        '0092_ANON9941969',
        '0093_ANON9728761',
        '0094_ANON8024204',
        '0095_ANON4189062',
        '0097_ANON5642073',
        '0103_ANON8583296',
        '0104_ANON7748630',
        '0105_ANON9883201',
        '0107_ANON4035085',
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
        '0136_ANON8096961',
        '0137_ANON8035619',
        '0138_ANON1747790',
        '0139_ANON2666319',
        '0140_ANON0899488',
        '0141_ANON8018038',
        '0142_ANON7090827',
        '0143_ANON9752849',
        '0144_ANON2255419',
        '0145_ANON0335209',
        '0146_ANON7414571',
        '0148_ANON9604223',
        '0149_ANON4712664',
        '0150_ANON5824292',
        '0152_ANON2411221',
        '0153_ANON5958718',
        '0155_ANON7828652',
        '0157_ANON9873056',
        '0159_ANON9720717',
        '0160_ANON3504149'
    ]
    uq_methods  = ['gaussian', 'lxo']  # 'gaussian' or # Leave-X-Out, where x are echo trains. for example L2O means that we leave out 2 echo trains.
    acc_factors = [3, 6] # Define the set of acceleration factors we care about.
    uq_metric   = 'std'  # 'std', 'cv', and more

    # === Roots ===
    reader_study_root = Path('/scratch/hb-pca-rad/projects/03_reader_set_v2')
    db_fpath_old             = Path('/scratch/p290820/datasets/003_umcg_pst_ksps/database/dbs/master_habrok_20231106_v2.db')               # References an OLDER version of the databases where the info could also just be fine that we are looking for
    db_fpath_new             = Path('/home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/databases/master_habrok_20231106_v2.db') # References the LATEST version of the databases where the info could also just be fine that we are looking for
    kspace_root_dir = Path('/scratch/p290820/datasets/003_umcg_pst_ksps')      # source_dir

    # === Configurable Params ===
    debug               = True
    do_round_sub_recons = True      # Round the sub-reconstructions to a certain number of decimals
    do_round_uq_map     = True      # Round the uncertainty map to a certain number of decimals
    do_norm             = True      # Normalize the reconstructions to [0, 1]
    do_post_processing  = True      # Post-processing, for each sub-reconstruction, such as k-space interpolation, flipping, and cropping
    save_subr_as_nifti  = True      # Save each sub-reconstructions as NIFTI files
    decimals            = 4         # Number of decimals to round to

    # === Level 1: UQ Method Loop ===
    for uq_method in uq_methods:
        LOGGER.info(f"\n🧠 Starting UQ Method: '{uq_method.upper()}'\n{'-'*60}")
        
        acc_roots = {
            3: Path(f"/scratch/hb-pca-rad/projects/04_uncertainty_quantification/{uq_method}/recons_3x"),
            6: Path(f"/scratch/hb-pca-rad/projects/04_uncertainty_quantification/{uq_method}/recons_6x")
        }

        for acc in acc_factors:
            LOGGER.info(f"🔄  Preparing data for R={acc}x | Method={uq_method}...")
            LOGGER.info(f"📂  R{acc}x root: {acc_roots[acc]}")
            
        LOGGER.info(f"\n🚀 Launching patient-level processing for {len(pat_ids)} patients using UQ method: '{uq_method}'\n")

        # === Level 2: Process Patients ===
        process_all_uncertainty_maps(
            uq_method           = uq_method,
            pat_ids             = pat_ids,
            acc_factors         = acc_factors,
            reader_study_root   = reader_study_root,
            acc_roots           = acc_roots,
            do_norm             = do_norm,
            do_post_processing  = do_post_processing,
            do_round_sub_recons = do_round_sub_recons,
            save_subr_as_nifti  = save_subr_as_nifti,
            kspace_root_dir     = kspace_root_dir,
            db_fpath_old        = db_fpath_old,
            do_round_uq_map     = do_round_uq_map,
            decimals            = decimals,
            debug               = debug,
            uq_metric           = uq_metric,
        )

        LOGGER.info(f"\n✅ Completed UQ Method: '{uq_method}' ({len(pat_ids)} patients processed)\n{'='*60}")
