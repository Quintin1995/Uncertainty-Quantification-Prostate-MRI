select
	pat_id,
	slice_idx,
	acc_factor,
	uq_method,
	region,
	mean_abs,
	mean_uq,
	spearman_corr
from 
	uq_vs_error_correlation_std
where
	uq_method = 'gaussian'
	AND
	region = 'slice'
	AND
	acc_factor = 6
order by
	spearman_corr desc;
	