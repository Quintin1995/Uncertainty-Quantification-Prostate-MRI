SELECT
    pat_id,
    acc_factor,
	ROUND(AVG(10000*mean_uq_slice), 1)   AS mean_uq_slice,
	ROUND(AVG(mean_abs_slice), 1)		 AS mean_abs_slice,
	ROUND(AVG(pearson_corr_slice), 3)    AS mean_corr_slice,
    ROUND(AVG(pearson_corr_prostate), 3) AS mean_corr_prostate,
    COUNT(*) AS nslices
FROM slice_level_uq_stats_debug
WHERE pearson_corr_slice IS NOT NULL OR pearson_corr_prostate IS NOT NULL
GROUP BY pat_id, acc_factor
ORDER BY pat_id, acc_factor;