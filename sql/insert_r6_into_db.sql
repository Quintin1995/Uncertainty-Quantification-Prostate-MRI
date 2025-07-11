INSERT INTO patients_uq (id, seq_id, anon_id, gaussian_id, recon_path, avg_acceleration)
SELECT
    id,
    seq_id,
    anon_id,
    0 AS gaussian_id,
    NULL AS recon_path,
    6 AS avg_acceleration
FROM patients_uq
WHERE rowid IN (
  SELECT MIN(rowid)
  FROM patients_uq
  GROUP BY id
);