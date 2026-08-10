# Release notes: production Variant A / RF v1

- Migrated the production evidence backend from the 577,118-row legacy snapshot to the
  frozen 280,917-row evidence-supported Variant A universe.
- Activated `mirassist_rf_variant_a_v1` and canonical `mirassist_score` loading.
- Preserved `mirassist_xgboost_score` unchanged for rollback; RF values are never stored there.
- Added model version, frozen global rank, within-miRNA percentile, and filtered/display rank plumbing.
- Updated planner retrieval and synthesis bundles to use model-agnostic score fields.
- Updated user-facing language to “miRAssist score”; technical views identify the RF model.
- Added safe zero/sparse result behavior with no Variant D fallback.
- Added non-destructive hosted database staging/cutover/rollback migrations and indexes.
- Added fixed staging/post-cutover regressions, release-fidelity checks, and rollback manifests.

See `docs/production_variant_a_rf_v1.md` for schema, limitations, score semantics, and rollback.
