import SimpleITK as sitk
import numpy as np
import sqlite3
import pandas as pd

from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
from scipy.stats import spearmanr

from assets.dicom_utils import resample_to_reference
from assets.uncertainty_quantification import apply_percentile_threshold
from assets.image_operations import apply_gaussian_blur_3d


# DOCUMENTATION
# we make a couple of table in an existing database
# 1. Summary Statistics to Store (Per Region & Acceleration)
# For both Uncertainty and Absolute Error, and per region (lesion & prostate):

# Metric	Use case
# mean	Average intensity — simple global indicator
# median	Robust to outliers
# max	Outlier/hallucination indicator
# std	Variation in signal (texture proxy)
# volume	Total #voxels in mask
# hotspot_volume_95p	#voxels > 95th percentile (global or per-patient)
# mean_above_95p	Mean of values in hotspot — severity of high-error/UQ areas
# percent_above_95p	% of mask occupied by hotspot
# Additionally for correlation analysis:

# pearson_corr between UQ and ABS (per-voxel inside lesion/prostate).

# spearman_corr if distributions are not normal.

# We do not store the whole images, just these features.

def get_combined_rois_array(pat_root: Path, r1_ref_image: sitk.Image, r1_arr: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    roi_fpaths = list(pat_root.glob("*_roi_*.mha"))
    roi_arrs_combined = np.zeros_like(r1_arr)

    # Check if any ROI files are found
    if len(roi_fpaths) == 0:
        print(f"\tNo ROIs found in {pat_root.name}.")
        return roi_arrs_combined, []

    for roi_fpath in roi_fpaths:
        roi_img = sitk.ReadImage(str(roi_fpath))
        roi_img_resampled = resample_to_reference(roi_img, r1_ref_image)
        roi_arr = sitk.GetArrayFromImage(roi_img_resampled)
        roi_arrs_combined += roi_arr
    slice_idxs_lesion = [i for i in range(len(roi_arrs_combined)) if np.sum(roi_arrs_combined[i]) > 0]
    print(f"\tCombined ROI {roi_fpath.name} has {len(slice_idxs_lesion)} slices with lesions. With idxs: {slice_idxs_lesion}")
    return roi_arrs_combined, slice_idxs_lesion


def compute_region_stats(abs_vals: np.ndarray, uq_vals: np.ndarray, prefix: str) -> Dict[str, Any]:
    stats = {
        # Absolute Error map
        f"mean_abs_{prefix}": np.mean(abs_vals),
        f"median_abs_{prefix}": np.median(abs_vals),
        f"min_abs_{prefix}": np.min(abs_vals),
        f"max_abs_{prefix}": np.max(abs_vals),
        f"std_abs_{prefix}": np.std(abs_vals),
        # Uncertainty map
        f"mean_uq_{prefix}": np.mean(uq_vals),
        f"median_uq_{prefix}": np.median(uq_vals),
        f"min_uq_{prefix}": np.min(uq_vals),
        f"max_uq_{prefix}": np.max(uq_vals),
        f"std_uq_{prefix}": np.std(uq_vals),
    }
    if abs_vals.size > 1 and uq_vals.size > 1:
        stats[f"pearson_corr_{prefix}"] = np.corrcoef(abs_vals, uq_vals)[0, 1]
        stats[f"spearman_corr_{prefix}"] = spearmanr(abs_vals, uq_vals).correlation
    else:
        stats[f"pearson_corr_{prefix}"] = None
        stats[f"spearman_corr_{prefix}"] = None
    return stats


def empty_region_stats(prefix: str) -> Dict[str, Any]:
    return {f"{s}_{prefix}": None for s in [
        "mean_abs", "median_abs", "min_abs", "max_abs", "std_abs",
        "mean_uq", "median_uq", "min_uq", "max_uq", "std_uq",
        "pearson_corr", "spearman_corr"
    ]}


def compute_slice_level_stats(
    pat_id: str,
    roots: Dict[Union[int, str], Path],
    acc_factors: List[int],
    do_blurring: bool = False,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Compute slice-level statistics for a single patient.

    Args:
        pat_id (str): Patient ID.
        roots (Dict[Union[int, str], Path]): Dictionary of paths for data.
        acc_factors (List[int]): Acceleration factors to process.
        debug (bool): If True, print debug information.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing slice-level statistics.
    """

    # Load R1 and Load lesion
    pat_root          = roots['reader_study'] / pat_id
    r1_img            = sitk.ReadImage(str(pat_root / f"{pat_id}_rss_target_dcml.mha"))
    r1_arr            = sitk.GetArrayFromImage(r1_img)
    roi_arr, les_idxs = get_combined_rois_array(pat_root, r1_img, r1_arr)
    print(f"\tr1_arr stats: max={np.max(r1_arr)}, min={np.min(r1_arr)}, mean={np.mean(r1_arr)}, median={np.median(r1_arr)}, std={np.std(r1_arr)}, shape={r1_arr.shape}")

    # Load prostate segmentation
    prostate_seg_root = roots['reader_study_segs'] / f"{pat_id}_mlseg_total_mr.nii.gz"
    prost_seg_arr     = sitk.GetArrayFromImage(sitk.ReadImage(str(prostate_seg_root))) # so this is a segmentation of many multi-label anatomical structures, where are interested in where it is the prostate=17
    prost_seg_arr     = np.where(prost_seg_arr == 17, 1, 0) # this is the prostate segmentation
    print(f"\tProstate segmentation stats: max={np.max(prost_seg_arr)}, min={np.min(prost_seg_arr)} std={np.std(prost_seg_arr)}, shape={prost_seg_arr.shape}")

    # Load R3 vSHARP Reconstruction --> compute Absolute Error and Load UQ map
    r3_arr            = sitk.GetArrayFromImage(sitk.ReadImage(str(pat_root / f"{pat_id}_VSharp_R3_recon_dcml.mha")))
    print(f"\tR3 reconstruction stats: max={np.max(r3_arr)}, min={np.min(r3_arr)}, mean={np.mean(r3_arr)}, median={np.median(r3_arr)}, std={np.std(r3_arr)}, shape={r3_arr.shape}")
    r3_abs_error_arr  = np.abs(r1_arr - r3_arr)
    # r3_abs_error_blu_arr = apply_gaussian_blur_3d(r3_abs_error_arr, sigma_xy=1.0, sigma_z=0.0)
    print(f"\tR3 Absolute Error stats: max={np.max(r3_abs_error_arr)}, min={np.min(r3_abs_error_arr)}, mean={np.mean(r3_abs_error_arr)}, median={np.median(r3_abs_error_arr)}, std={np.std(r3_abs_error_arr)}, shape={r3_abs_error_arr.shape}")
    r3_uq_map_arr     = sitk.GetArrayFromImage(sitk.ReadImage(str(roots['R3'] / pat_id / f"uq_map_R3_gm25.nii.gz")))
    # r3_uq_map_blu_arr = apply_gaussian_blur_3d(r3_uq_map_arr, sigma_xy=1.0, sigma_z=0.0)
    print(f"\tR3 UQ map stats: max={np.max(r3_uq_map_arr)}, min={np.min(r3_uq_map_arr)}, mean={np.mean(r3_uq_map_arr)}, median={np.median(r3_uq_map_arr)}, std={np.std(r3_uq_map_arr)}, shape={r3_uq_map_arr.shape}")

    # Load R6 Reconstruction --> compute Absolute Error and Load UQ map
    r6_arr            = sitk.GetArrayFromImage(sitk.ReadImage(str(pat_root / f"{pat_id}_VSharp_R6_recon_dcml.mha")))
    print(f"\tR6 reconstruction stats: max={np.max(r6_arr)}, min={np.min(r6_arr)}, mean={np.mean(r6_arr)}, median={np.median(r6_arr)}, std={np.std(r6_arr)}, shape={r6_arr.shape}")
    r6_abs_error_arr  = np.abs(r1_arr - r6_arr)
    # r6_abs_error_blu_arr = apply_gaussian_blur_3d(r6_abs_error_arr, sigma_xy=1.0, sigma_z=0.0)
    print(f"\tR6 Absolute Error stats: max={np.max(r6_abs_error_arr)}, min={np.min(r6_abs_error_arr)}, mean={np.mean(r6_abs_error_arr)}, median={np.median(r6_abs_error_arr)}, std={np.std(r6_abs_error_arr)}, shape={r6_abs_error_arr.shape}")
    r6_uq_map_arr     = sitk.GetArrayFromImage(sitk.ReadImage(str(roots["R6"] / pat_id / f"uq_map_R6_gm25.nii.gz")))
    # r6_uq_map_blu_arr = apply_gaussian_blur_3d(r6_uq_map_arr, sigma_xy=1.0, sigma_z=0.0)
    print(f"\tR6 UQ map stats: max={np.max(r6_uq_map_arr)}, min={np.min(r6_uq_map_arr)}, mean={np.mean(r6_uq_map_arr)}, median={np.median(r6_uq_map_arr)}, std={np.std(r6_uq_map_arr)}, shape={r6_uq_map_arr.shape}")

    if do_blurring:
        r3_abs_error_arr = apply_gaussian_blur_3d(r3_abs_error_arr, sigma_xy=1.0, sigma_z=0.0)
        r3_uq_map_arr    = apply_gaussian_blur_3d(r3_uq_map_arr, sigma_xy=1.0, sigma_z=0.0)
        r6_abs_error_arr = apply_gaussian_blur_3d(r6_abs_error_arr, sigma_xy=1.0, sigma_z=0.0)
        r6_uq_map_arr    = apply_gaussian_blur_3d(r6_uq_map_arr, sigma_xy=1.0, sigma_z=0.0)

    # ---- MAIN LOOP ---- over, Acceleration Factor and Slices
    all_slice_stats = []
    for acc in acc_factors:
        abs_arr = r3_abs_error_arr if acc == 3 else r6_abs_error_arr
        uq_arr  = r3_uq_map_arr    if acc == 3 else r6_uq_map_arr
        
        for i in range(r1_arr.shape[0]):
            slice_stat = {
                "pat_id": pat_id,
                "slice_idx": i,
                "acc_factor": acc,
            }
            # --- Whole slice stats ---
            abs_vals = abs_arr[i].flatten()
            uq_vals  = uq_arr[i].flatten()
            slice_stat.update(compute_region_stats(abs_vals, uq_vals, "slice"))

            # ----- Prostate stats -----
            pr_mask = prost_seg_arr[i] == 1
            if np.any(pr_mask):
                abs_vals = abs_arr[i][pr_mask]
                uq_vals  = uq_arr[i][pr_mask]
                slice_stat.update(compute_region_stats(abs_vals, uq_vals, "prostate"))
            else:
                slice_stat.update(empty_region_stats("prostate"))

            # --- Lesion stats ---
            ls_mask = roi_arr[i] > 0
            if np.any(ls_mask):
                abs_vals = abs_arr[i][ls_mask]
                uq_vals  = uq_arr[i][ls_mask]
                slice_stat.update(compute_region_stats(abs_vals, uq_vals, "lesion"))
            else:
                slice_stat.update(empty_region_stats("lesion"))

            all_slice_stats.append(slice_stat)

    if VERBOSE:
        print(f"\n🔎 {len(all_slice_stats)} slice-level rows collected for patient {pat_id}")
        for row in all_slice_stats:  # print first few
            print(f"  ➤ Slice {row['slice_idx']} @ R={row['acc_factor']}: "
                f"UQ_lesion_mean={row['mean_uq_lesion']}, ABS_prost_max={row['max_abs_prostate']}, Whole Slice mean UQ={row['mean_uq_slice']}")

    return all_slice_stats


def process_patients_and_store_stats(
    pat_ids: List[str],
    roots: Dict[Union[int, str], Path],
    acc_factors: List[int],
    db_fpath: Path,
    table_name: str = "uq_vs_abs_stats",
    do_blurring: bool = False,
    debug: bool = False,
):
    """
    Process multiple patients, compute slice-level statistics, and store them in a SQLite database.

    Args:
        pat_ids (List[str]): List of patient IDs to process.
        roots (Dict[Union[int, str], Path]): Dictionary of paths for data.
        acc_factors (List[int]): Acceleration factors to process.
        db_fpath (Path): Path to the SQLite database file.
        table_name (str): Name of the table to store the statistics.
        debug (bool): If True, print debug information.

    Returns:
        None
    """
    all_stats = []

    # Process each patient
    for idx, pat_id in enumerate(pat_ids):
        print(f"\n{idx + 1}/{len(pat_ids)} Processing patient {pat_id}...")
        slice_stats = compute_slice_level_stats(
            pat_id      = pat_id,
            roots       = roots,
            acc_factors = acc_factors,
            do_blurring = do_blurring,
            debug       = debug,
        )
        all_stats.extend(slice_stats)   # extend is not append. Extend does: [1, 2] [3, 4] becomes [1,2,3,4] instead of [1, 2, [3, 4]]
    stats_df = pd.DataFrame(all_stats)

    # Store in SQLite database
    print(f"\nStoring results in SQLite database: {db_fpath}")
    with sqlite3.connect(str(db_fpath)) as conn:
        stats_df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"Data successfully stored in table '{table_name}'.")

    if debug:
        print(f"\nPreview of stored data:")
        print(stats_df.head())


def create_table_if_not_exists(db_fpath: Path, table_name: str, debug: bool = False):
    """
    Create a new table in the SQLite database if it does not already exist.
    If debug is True, '_debug' is appended to the table name.

    Args:
        db_fpath (Path): Path to the SQLite database file.
        table_name (str): Name of the table to create.
        debug (bool): If True, append '_debug' to the table name.

    Returns:
        str: The final table name used.
    """
    # Append '_debug' to the table name if debug is True
    final_table_name = f"{table_name}_debug" if debug else table_name

    # Define the SQL command to create the table
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {final_table_name} (
        pat_id TEXT,
        slice_idx INTEGER,
        acc_factor INTEGER,
        mean_abs_slice REAL,
        median_abs_slice REAL,
        min_abs_slice REAL,
        max_abs_slice REAL,
        std_abs_slice REAL,
        mean_uq_slice REAL,
        median_uq_slice REAL,
        min_uq_slice REAL,
        max_uq_slice REAL,
        std_uq_slice REAL,
        pearson_corr_slice REAL,
        spearman_corr_slice REAL,
        mean_abs_prostate REAL,
        median_abs_prostate REAL,
        min_abs_prostate REAL,
        max_abs_prostate REAL,
        std_abs_prostate REAL,
        mean_uq_prostate REAL,
        median_uq_prostate REAL,
        min_uq_prostate REAL,
        max_uq_prostate REAL,
        std_uq_prostate REAL,
        pearson_corr_prostate REAL,
        spearman_corr_prostate REAL,
        mean_abs_lesion REAL,
        median_abs_lesion REAL,
        min_abs_lesion REAL,
        max_abs_lesion REAL,
        std_abs_lesion REAL,
        mean_uq_lesion REAL,
        median_uq_lesion REAL,
        min_uq_lesion REAL,
        max_uq_lesion REAL,
        std_uq_lesion REAL,
        pearson_corr_lesion REAL,
        spearman_corr_lesion REAL
    );
    """

    # Connect to the database and execute the SQL command
    try:
        with sqlite3.connect(str(db_fpath)) as conn:
            cursor = conn.cursor()
            cursor.execute(create_table_sql)
            conn.commit()
        print(f"Table '{final_table_name}' created successfully (or already exists).")
    except sqlite3.Error as e:
        print(f"SQLite error while creating table '{final_table_name}': {e}")
        raise

    return final_table_name



if __name__ == '__main__':

    # All patient IDs to consider for Uncertainty Quantification
    pat_ids = [
        '0003_ANON5046358',
        '0004_ANON9616598',
        '0005_ANON8290811',
        '0006_ANON2379607',
        '0007_ANON1586301',
        '0008_ANON8890538',
        '0010_ANON7748752',
        '0011_ANON1102778',
        '0012_ANON4982869',
        '0013_ANON7362087',
        '0014_ANON3951049',
        '0015_ANON9844606',
        '0018_ANON9843837',
        '0019_ANON7657657',
        '0020_ANON1562419',
        '0021_ANON4277586',
        '0023_ANON6964611',
        '0024_ANON7992094',
        '0026_ANON3620419',
        '0027_ANON9724912',
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
        '0079_ANON8828023',
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

    roots = {
        'reader_study':      Path('/scratch/hb-pca-rad/projects/03_reader_set_v2'),
        'reader_study_segs': Path('/scratch/hb-pca-rad/projects/03_reader_set_v2/segs'),
        'R3':                Path(f"/scratch/hb-pca-rad/projects/04_uncertainty_quantification/gaussian/recons_{3}x"),
        'R6':                Path(f"/scratch/hb-pca-rad/projects/04_uncertainty_quantification/gaussian/recons_{6}x"),
        'kspace_root':       Path('/scratch/p290820/datasets/003_umcg_pst_ksps'),
        'db_fpath_old':      Path('/scratch/p290820/datasets/003_umcg_pst_ksps/database/dbs/master_habrok_20231106_v2.db'),                 # References an OLDER version of the databases where the info could also just be fine that we are looking for
        'db_fpath_new':      Path('/home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/databases/master_habrok_20231106_v2.db'),   # References the LATEST version of the databases where the info could also just be fine that we are looking for
    }
    do_blurring   = True
    acc_factors   = [3, 6] # Define the set of acceleration factors we care about.
    DEBUG         = False
    VERBOSE       = True
    
    table_name = create_table_if_not_exists(
        db_fpath   = roots["db_fpath_new"],
        table_name = "slice_level_uq_stats",
        debug      = DEBUG,
    )

    # Process patients and store results
    process_patients_and_store_stats(
        pat_ids     = pat_ids,
        roots       = roots,
        acc_factors = acc_factors,
        db_fpath    = roots["db_fpath_new"],
        table_name  = table_name,
        do_blurring = do_blurring,
        debug       = DEBUG,
    )