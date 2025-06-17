SELECT 
    acc_factor,
    uq_method,
    region,
    round(AVG(mean_abs), 2) AS avg_abs,
    round(AVG(mean_uq ), 6) AS avg_mean_uq,
	round(AVG(std_uq  ), 6) AS avg_std_uq
FROM uq_vs_error_correlation_std
GROUP BY acc_factor, uq_method, region;
