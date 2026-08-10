-- Stage the approved Variant A/RF v1 release in a new table.
-- This migration is intentionally non-destructive: it never renames, truncates,
-- or drops public.mirassist_evidence_pairs.

create table if not exists public.mirassist_evidence_variant_a_rf_v1
(like public.mirassist_evidence_pairs including defaults including constraints including storage including comments);

alter table public.mirassist_evidence_variant_a_rf_v1
  add column if not exists mirassist_model_score double precision,
  add column if not exists mirassist_model_version text,
  add column if not exists mirassist_score_rank_within_mirna bigint,
  add column if not exists mirassist_score_percentile_within_mirna double precision;

comment on table public.mirassist_evidence_variant_a_rf_v1 is
  'mirassist_evidence_variant_a_rf_v1: frozen 280,917-row evidence-supported Variant A universe scored by mirassist_rf_variant_a_v1';
comment on column public.mirassist_evidence_variant_a_rf_v1.mirassist_model_score is
  'Raw uncalibrated random-forest positive-class vote fraction used solely as a relative prioritization score; no biological probability interpretation.';
comment on column public.mirassist_evidence_variant_a_rf_v1.mirassist_xgboost_score is
  'Legacy score field retained unchanged; RF scores must never be written here.';

create unique index if not exists uq_mirassist_variant_a_rf_v1_biological_key
  on public.mirassist_evidence_variant_a_rf_v1
  (mirna_name_normalized, gene_symbol_normalized, transcript_id);

create index if not exists idx_mirassist_variant_a_rf_v1_mirna_score
  on public.mirassist_evidence_variant_a_rf_v1
  (mirna_name_normalized, mirassist_model_score desc,
   overall_evidence_support_percentile desc, evidence_family_count desc,
   gene_symbol_normalized, transcript_id, evidence_row_id);

create index if not exists idx_mirassist_variant_a_rf_v1_gene_score
  on public.mirassist_evidence_variant_a_rf_v1
  (gene_symbol_normalized, mirassist_model_score desc, mirna_name_normalized, transcript_id);

create index if not exists idx_mirassist_variant_a_rf_v1_novel
  on public.mirassist_evidence_variant_a_rf_v1
  (mirna_name_normalized, mirtarbase_known_positive, mirassist_model_score desc);

create index if not exists idx_mirassist_variant_a_rf_v1_common_filters
  on public.mirassist_evidence_variant_a_rf_v1
  (support_count, support_targetscan, support_encori, BRCA_support_tcga, COAD_support_tcga, PRAD_support_tcga);

create table if not exists public.mirassist_production_metadata (
  singleton boolean primary key default true check (singleton),
  schema_version text not null,
  candidate_universe_version text not null,
  model_version text not null,
  score_field text not null,
  active_table text not null,
  row_count bigint not null,
  source_sha256 text not null,
  cutover_at timestamptz
);

insert into public.mirassist_production_metadata
  (singleton, schema_version, candidate_universe_version, model_version,
   score_field, active_table, row_count, source_sha256, cutover_at)
values
  (true, 'mirassist_evidence_variant_a_rf_v1', 'variant_a',
   'mirassist_rf_variant_a_v1', 'mirassist_model_score',
   'public.mirassist_evidence_variant_a_rf_v1', 280917,
   '2fc1b25af55c22c7e44e4587ac586942c0b5f3eb47afe0e36ccf5cab0512a9ee', null)
on conflict (singleton) do update set
  schema_version = excluded.schema_version,
  candidate_universe_version = excluded.candidate_universe_version,
  model_version = excluded.model_version,
  score_field = excluded.score_field,
  active_table = excluded.active_table,
  row_count = excluded.row_count,
  source_sha256 = excluded.source_sha256;
