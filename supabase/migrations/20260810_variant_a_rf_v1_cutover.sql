-- Run only after the staged table passes all release-fidelity checks.
begin;
create or replace view public.mirassist_evidence_active as
  select * from public.mirassist_evidence_variant_a_rf_v1;
update public.mirassist_production_metadata
set active_table = 'public.mirassist_evidence_variant_a_rf_v1',
    cutover_at = current_timestamp
where singleton;
commit;
