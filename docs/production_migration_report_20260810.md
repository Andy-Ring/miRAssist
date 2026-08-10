# Production migration report: Variant A / RF v1

Cutover timestamp: `2026-08-10T15:50:29Z`

## Before migration

- Git branch/base commit: `main` / `da02e3571c71856ec2c956f46ece3b60c7e083c6`
- Runtime default: GitHub v0.0.1 legacy snapshot
- Local rollback source: `data/processed/mirassist_evidence_pairs.parquet`
- Legacy shape: 577,118 rows × 126 columns
- Legacy Parquet SHA-256: `04e3ddb12eea5b99c6ed60715a178993085e7c17d1fba1354989d2ead77e7e61`
- Legacy CSV.GZ SHA-256: `db386c8c73ca5ec2b50fac89c39a2383496c403280210bdf6f4b658de1d86332`
- Legacy known-positive rows: 5,241
- Legacy persisted score: `mirassist_xgboost_score`
- Legacy hosted table name: `public.mirassist_evidence_pairs`
- No database credentials/URL or deployment CLI were configured in this workspace.

The repository had substantial unrelated uncommitted scientific-pipeline work before
migration. Those files were not overwritten or incorporated into migration edits.

## Backup

Complete rollback files, schema, Git state, configuration, and checksums are under:

`outputs/production_migration/20260810T152626Z/rollback/`

Primary manifest:
`outputs/production_migration/20260810T152626Z/rollback/rollback_manifest.json`.

## Approved release verification

- Scored table: 280,917 rows × 130 columns
- Table SHA-256: `2fc1b25af55c22c7e44e4587ac586942c0b5f3eb47afe0e36ccf5cab0512a9ee`
- Model SHA-256: `c765ff90ef841d05e976f8948318cd644f60bbd94c1e3466eca197f35dceeb94`
- All 34 release checksum entries passed.
- All 126 original columns matched the frozen premodel table exactly.
- Biological keys were unique; RF scores were complete/finite; model version was uniform.
- Stored ranks and percentiles reproduced exactly.
- Known-positive count was 2,583.
- Every row passed Variant A eligibility; Variant D rows: 0.
- Held-out AUROC: 0.8462379826608851.
- Held-out PR-AUC: 0.13475229166015865.

Verification report:
`outputs/production_migration/20260810T152626Z/staging/release_fidelity.json`.

## Cutover

- Active local table: `data/processed/mirassist_evidence_variant_a_rf_v1.parquet`
- Active table checksum: identical to approved source (`2fc1b25a…`).
- Active schema: `mirassist_evidence_variant_a_rf_v1`.
- Active universe: Variant A.
- Active model: `mirassist_rf_variant_a_v1`.
- Persisted active score: `mirassist_model_score`.
- Runtime canonical score: `mirassist_score`.
- Hosted versioned table DDL: `public.mirassist_evidence_variant_a_rf_v1`.

The source scored release remained unchanged during installation. The local production
table was installed as a byte-identical copy after all guardrails passed.

## Validation

Staging regression report:
`outputs/production_migration/20260810T152626Z/staging/regression_smoke.json`.

It covers high-, median-, low-, and zero-coverage miRNAs; top-k larger than the candidate
count; pathway filtering; BRCA/COAD/PRAD contexts; novel mode; gene→miRNA retrieval;
synthesis-bundle construction; CSV export; and debug metadata. Returned evidence row IDs,
RF scores, and global ranks matched the frozen release.

Post-cutover results are recorded at:
`outputs/production_migration/20260810T152626Z/post_cutover/production_smoke.json`.

The Streamlit application started successfully on a temporary local port using the
production-default configuration and stopped normally after the bounded startup check.
All non-training test modules passed when run in memory-isolated processes: 169 passed
and 2 skipped. The historical `test_variant_a_model_training_evaluation.py` module could
not collect because the pipeline environment does not contain its legacy `xgboost`
dependency. That dependency was not installed and the training test was not run because
this migration prohibits retraining. Release-fidelity and RF scientific-review tests are
included among the 169 passing tests.

## Code and database changes

- Added canonical model-agnostic loader with new-score precedence, legacy fallback, and conflict detection.
- Replaced active XGBoost-specific ranking assumptions with canonical score ordering.
- Preserved frozen global rank and added separate filtered/display rank.
- Updated novel mode, empty-result wording, UI terminology, exports, synthesis payloads, and debug metadata.
- Added versioned PostgreSQL staging, cutover, rollback, metadata, and index migrations.
- Updated stale evidence-family tests and the score-valued `learned_score_used` smoke assertion.
- Updated stale regressions that expected phenotype directionality to override approved
  RF ordering or expected an unscored candidate-generation artifact to be accepted as a
  production ranking table. Directionality annotation/filtering remains intact; unscored
  production input now fails closed.

## Immutability, rollback, and LLM-validation freeze

- Legacy source and backup Parquet SHA-256 both remain
  `04e3ddb12eea5b99c6ed60715a178993085e7c17d1fba1354989d2ead77e7e61`.
- Approved source and installed production Parquet SHA-256 both remain
  `2fc1b25af55c22c7e44e4587ac586942c0b5f3eb47afe0e36ccf5cab0512a9ee`.
- The legacy table/model assets and approved RF model remain retained.
- Pointer-only rollback instructions are at
  `outputs/production_migration/20260810T152626Z/rollback/rollback_instructions.md`.
- The cutover manifest is at
  `outputs/production_migration/20260810T152626Z/cutover/cutover_manifest.json`.
- The frozen environment for the forthcoming (not executed) LLM validation is at
  `outputs/production_migration/20260810T152626Z/llm_validation/llm_validation_environment_manifest.json`.
  It records active data/model fields and checksums, planner/synthesis model names,
  prompt checksums, the application base commit, and per-file migration checksums.

## Deployment deviation

The active application in this workspace now uses the new local production table and has
passed local post-cutover smoke tests. No live PostgreSQL/Cloud Run/Posit deployment was
mutated because this environment exposes neither `DATABASE_URL` nor `gh`/`gcloud`/Posit
deployment tooling. The hosted SQL and deployment configuration are prepared and
reversible, but an external hosted-service cutover requires the deployment environment.

The Git base commit remains `da02e357…`; migration changes are present in the working tree
and are frozen by per-file checksums in the LLM-validation environment manifest rather
than by an unauthorized Git commit.
