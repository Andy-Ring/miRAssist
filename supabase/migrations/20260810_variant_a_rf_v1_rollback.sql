-- Reversible pointer rollback. The Variant A table is deliberately retained.
begin;
create or replace view public.mirassist_evidence_active as
  select * from public.mirassist_evidence_pairs;
update public.mirassist_production_metadata
set active_table = 'public.mirassist_evidence_pairs',
    schema_version = 'legacy_126_column',
    candidate_universe_version = 'legacy_unspecified',
    model_version = 'legacy_xgboost_unspecified',
    score_field = 'mirassist_xgboost_score',
    row_count = 577118,
    source_sha256 = '04e3ddb12eea5b99c6ed60715a178993085e7c17d1fba1354989d2ead77e7e61',
    cutover_at = current_timestamp
where singleton;
commit;
