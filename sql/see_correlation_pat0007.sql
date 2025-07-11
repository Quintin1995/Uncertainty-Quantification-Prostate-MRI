SELECT
    slice_idx,
    acc_factor,
    ROUND(pearson_corr_lesion, 3)     AS lesion_corr,
    ROUND(pearson_corr_prostate, 3)   AS prostate_corr,
    ROUND(pearson_corr_slice, 3)      AS slice_corr,
    ROUND(mean_uq_lesion, 3)          AS uq_lesion_mean,
    ROUND(max_abs_lesion, 3)          AS abs_lesion_max
FROM slice_level_uq_stats_debug
WHERE pat_id = '0007_ANON1586301'
ORDER BY acc_factor, slice_idx;
