alter table mirassist_evidence_pairs
add column if not exists learned_score_xgb_raw_v1 double precision,
add column if not exists learned_score_xgb_raw_nomissing_v1 double precision,
add column if not exists learned_score_model_version text,
add column if not exists learned_score_feature_set text,
add column if not exists learned_score_updated_at timestamptz;

create index if not exists idx_mirassist_evidence_pairs_learned_xgb_raw_v1
on mirassist_evidence_pairs (mirna_name_norm, learned_score_xgb_raw_v1 desc);

create index if not exists idx_mirassist_evidence_pairs_gene_learned_xgb_raw_v1
on mirassist_evidence_pairs (gene_symbol_norm, learned_score_xgb_raw_v1 desc);
