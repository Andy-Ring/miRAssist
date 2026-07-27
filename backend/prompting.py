"""
Prompt templates + bundle builder for miRAssist.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from backend.config import get_default_result_count, use_mirtarbase_evidence


SYSTEM_PROMPT = """You are miRAssist, a scientific assistant that helps prioritize miRNA-mRNA interactions for experimental follow-up.

Your primary goal is to present results clearly, consistently, and accurately, using only the provided evidence.

Hard rules:
- Use ONLY the provided evidence cards and the user's question/context. Do not invent evidence.
- The backend has already filtered and ranked the candidate pool. Preserve the provided candidate order.
- Do not re-rank candidates unless the user explicitly asks you to ignore backend ranking.
- Output EXACTLY the requested number of ranked items when enough candidates are available.
- Never list the same gene or miRNA more than once.
- Do not invent gene-to-phenotype links.
- Do not invent pathway membership.
- Do not infer pathway directionality unless the pathway name explicitly supports it.
- Do not imply that the miRNA directly activates or induces target gene expression.
- Do not calculate percentiles yourself; use the backend-provided raw values and support percentiles.
- Do not invent feature strengths; use only the backend-provided raw values, support percentiles, and caveats.
- Use the backend-provided evidence_support_count exactly as given.
- Percentiles are interpretation aids for the associated feature or evidence family, not separate evidence.
- Do not count a raw feature and its percentile annotation separately.
- Evidence support count is based on distinct evidence families, not individual columns.
- Use at most one or two primary metrics per evidence family unless the card explicitly marks additional metrics as nonredundant.
- Do not list best/mean/max variants of the same evidence type as separate key evidence unless the card makes that necessary.
- Do not include low or absent evidence as a key evidence line unless it is an important caveat.
- Evidence support count measures breadth, not strength.
- A candidate with fewer evidence families may still be more compelling if those families are strong.
- A candidate with more categories may still be exploratory if the values are weak or typical.
- When explaining rankings, mention both evidence breadth and evidence strength.
- Do not call a candidate "strongest" solely because it has more categories.
- If the backend provides `overall_priority_tier`, use it.
- Do not describe computational predictions as experimental validation.

Evidence interpretation rules:
- The six major evidence families are sequence complementarity, thermodynamic stability, sequence conservation, target site accessibility, functional binding, and functional repression.
- More negative RNAhybrid MFE means stronger predicted binding support.
- More negative TargetScan context score means stronger sequence-conservation support.
- Higher accessibility probabilities mean the target region is more accessible.
- Higher CLIP or ENCORI values mean stronger binding support.
- More negative TCGA Spearman rho means stronger repression-consistent anticorrelation.
- CLIP supports binding evidence, not necessarily repression.
- TCGA context evidence is context-specific repression support, not direct binding evidence.
- Pathway evidence only means the candidate passed the deterministic pathway filter or has explicit pathway names in the card.
- If the backend provides a target-role interpretation, describe it as a candidate interpretation that is consistent with miRNA-mediated repression and grounded in pathway annotations.
- Phrase directional context as "consistent with" or "candidate target interpretation", not as proof that the target gene caused the phenotype.

Novel mode rules:
- In novel mode, known curated interactions excluded by the backend must NOT appear in the ranked list.
- Known interactions may be mentioned only as background context, not as novel candidates.

Required output format:

## Interpretation
- A short paragraph or 2-4 bullets stating:
  - what miRNA/gene is being queried
  - whether the task is miRNA->targets or gene->miRNAs
  - whether novel mode is on/off
  - whether a cancer context was used
  - whether a pathway/phenotype filter was applied
  - if pathway filtering was applied, that results are restricted to genes in selected pathways

## Results
- Present candidates in the exact order provided by the backend evidence cards.
- If fewer than the requested number of candidates were provided, say exactly: "Fewer than N candidates passed the filters." replacing N with the requested count.
- If at least the requested number of candidates were provided, do not include any "fewer than" sentence.
- For each result use this format:

### 1. GENE_OR_MIRNA
- **Overall priority:** Strong / Moderate / Exploratory / Context-limited / Conflicting context
- **Evidence support count:** X families
- **Evidence families:** sequence_complementarity; thermodynamic_stability; sequence_conservation; target_site_accessibility; functional_binding; functional_repression
- **Pathways:** pathway names if pathway filtering was applied, otherwise "Not pathway-filtered"
- **Predicted miRNA effect on target:** increased / decreased / unknown
- **Target role evidence:** explicit positive- or negative-regulator pathway annotations, or "Absent"
- **Directional consistency:** consistent / conflicting / role evidence absent / unknown
- **Key pieces of evidence:**
  - grouped by evidence family, with one or two primary metrics per family and percentiles treated only as modifiers
- **Interpretation:** 2-4 sentences grounded only in the evidence card, explicitly distinguishing evidence breadth from evidence strength

## Final recommendation
- A short paragraph saying which 1-3 candidates are best for follow-up and why.

Keep language professional, direct, and readable.
"""


def get_system_prompt() -> str:
    prompt = SYSTEM_PROMPT
    if use_mirtarbase_evidence():
        return prompt

    prompt = prompt.replace(
        "- In novel mode, known curated interactions excluded by the backend must NOT appear in the ranked list.\n",
        "",
    )
    return prompt


def build_user_prompt(
    *,
    user_question: str,
    direction: str,
    cancer: Optional[str] = None,
    cancer_name: Optional[str] = None,
    novel: bool = False,
    phenotype_keywords: Optional[List[str]] = None,
    pathway_keywords: Optional[List[str]] = None,
    pathway_selection: Optional[Dict[str, Any]] = None,
    phenotype_context: Optional[Dict[str, Any]] = None,
    target_role_inference: Optional[Dict[str, Any]] = None,
    cards: List[Dict[str, Any]],
    needs_clarification: Optional[List[str]] = None,
    requested_results: Optional[int] = None,
    retrieval_diagnostics: Optional[Dict[str, Any]] = None,
) -> str:
    phenotype_keywords = phenotype_keywords or []
    pathway_keywords = pathway_keywords or []
    pathway_selection = pathway_selection or {}
    phenotype_context = phenotype_context or {}
    target_role_inference = target_role_inference or {}
    needs_clarification = needs_clarification or []
    retrieval_diagnostics = retrieval_diagnostics or {}

    if direction == "mirna_to_targets":
        task = "Identify and rank target genes regulated by the miRNA."
        output_item = "genes"
        direction_label = "miRNA -> targets"
    elif direction == "gene_to_mirnas":
        task = "Identify and rank candidate miRNA regulators of the queried gene."
        output_item = "miRNAs"
        direction_label = "gene -> miRNAs"
    else:
        task = "Identify and rank relevant candidates from the evidence."
        output_item = "candidates"
        direction_label = "unknown"

    ctx_lines: List[str] = []
    if cancer_name or cancer:
        ctx_lines.append(f"- Cancer context: {cancer_name if cancer_name else cancer}")
    else:
        ctx_lines.append("- Cancer context: No specific cancer context requested. TCGA fields may still appear as functional repression evidence.")
    if phenotype_keywords:
        ctx_lines.append(f"- Phenotype keywords: {', '.join(phenotype_keywords)}")
    if pathway_keywords:
        ctx_lines.append(f"- Pathway keywords: {', '.join(pathway_keywords)}")
    if pathway_selection.get("enabled"):
        selected_names = [
            item.get("pathway_name", "")
            for item in (pathway_selection.get("selected_pathways") or [])
            if item.get("pathway_name")
        ]
        if selected_names:
            ctx_lines.append(f"- Selected pathways: {', '.join(selected_names[:8])}")
        if pathway_selection.get("warnings"):
            ctx_lines.append(f"- Pathway warnings: {'; '.join(pathway_selection.get('warnings') or [])}")
    if target_role_inference.get("enabled"):
        assumption = str(target_role_inference.get("assumption") or "").strip()
        reasoning = str(target_role_inference.get("reasoning") or "").strip()
        if assumption:
            ctx_lines.append(f"- Biological assumption: {assumption}")
        if reasoning:
            ctx_lines.append(f"- Target-role interpretation: {reasoning}")
        ctx_lines.append(
            "- Expected target expression change: "
            + str(target_role_inference.get("expected_target_expression_change") or "unknown")
        )
    if pathway_selection.get("enabled") and target_role_inference.get("enabled"):
        phenotype = phenotype_context.get("phenotype")
        expected_role = target_role_inference.get("expected_target_effect_on_phenotype")
        if phenotype and expected_role in {"positive_regulator", "negative_regulator"}:
            ctx_lines.append(
                "- Pathway grounding: directional pathway filtering was based on deterministic pathway annotations "
                f"for {expected_role.replace('_', ' ')} genes in {phenotype}, not on invented gene memberships."
            )
    if novel:
        ctx_lines.append("- Mode: NOVEL (known curated interactions excluded by the backend should not be ranked as novel)")
    if needs_clarification:
        ctx_lines.append(f"- Ambiguities noted by planner: {', '.join(needs_clarification)}")
    if retrieval_diagnostics.get("user_notes"):
        ctx_lines.append(
            "- Retrieval notes: " + "; ".join([str(item) for item in (retrieval_diagnostics.get("user_notes") or []) if str(item).strip()])
        )
    query_entity = ""
    if direction == "gene_to_mirnas" and cards:
        query_entity = str(cards[0].get("gene_symbol") or "").strip()
    elif direction == "mirna_to_targets" and cards:
        query_entity = str(cards[0].get("mirna_name") or "").strip()
    if query_entity:
        ctx_lines.append(f"- Resolved query: {query_entity}; task is {direction_label}.")

    available_n = len(cards)
    top_n = int(requested_results or available_n)
    if top_n <= 0:
        top_n = available_n
    instr = f"""Task:
{task}

Requirements:
- Requested ranked results: {top_n}
- Available evidence cards: {available_n}
- Resolved task type: {direction_label}
- Preserve the available evidence card order; it is the backend ranking order.
- Return EXACTLY {top_n} UNIQUE ranked {output_item} unless fewer than {top_n} candidates passed the filters.
- Use only the evidence cards below; do not invent extra support.
- Do not use unsupported intuition or external knowledge when ranking.
- Use the backend-provided evidence support count, evidence families, raw values, support percentiles, and caveats.
- Use the backend-provided evidence_support_count exactly as given.
- Percentiles are interpretation aids for the associated feature or family, not separate evidence.
- Do not count raw values and percentiles separately.
- Use at most one or two primary metrics per evidence family unless the card explicitly marks additional metrics as nonredundant.
- Keep "Key pieces of evidence" compact and grouped by family.
- Evidence support count means breadth, not strength.
- A candidate with fewer evidence families may still rank highly if those families are strong.
- A candidate with more categories may still be lower confidence if most values are weak or typical.
- Do not invent percentiles, pathway membership, or gene-phenotype links.
- Do not claim a candidate belongs to a pathway unless the card explicitly says it passed the pathway filter or names the pathways.
- If directional phenotype logic is provided, label it as an assumption based on typical miRNA-mediated repression and explain it in plain language, not as proof of causality.
- For every candidate, report the predicted miRNA effect on target expression, explicit positive/negative-regulator evidence, directional consistency, and whether role evidence is absent or conflicting.
- For gene -> miRNAs tasks, describe results as candidate miRNA regulators of the queried gene or miRNAs predicted to target/regulate that gene.
- For gene -> miRNAs tasks, do not describe the gene as a miRNA and do not call the returned miRNAs "targets" of the gene.
- Use the required output structure from the system prompt.

User question:
{user_question.strip()}

Context:
{chr(10).join(ctx_lines) if ctx_lines else "- (none provided)"}

Evidence cards:
"""

    if not cards:
        no_candidate_lines = [
            "No evidence cards were produced because backend retrieval returned no candidates after filtering.",
            f"Final shortlist size: {retrieval_diagnostics.get('n_final_shortlist', 0)}",
        ]
        if retrieval_diagnostics.get("warnings"):
            no_candidate_lines.append(
                "Backend diagnostics: " + "; ".join(retrieval_diagnostics.get("warnings") or [])
            )
        return instr + "\n\n" + "\n".join(no_candidate_lines)

    card_blocks: List[str] = []
    for index, card in enumerate(cards, start=1):
        raw_key_values = card.get("raw_key_values") or {}
        raw_value_line = "None"
        if raw_key_values:
            raw_value_line = "; ".join(f"{key}={value}" for key, value in raw_key_values.items())
        pathway_names = card.get("pathway_names") or []
        evidence_categories_present = card.get("evidence_families_present") or card.get("evidence_categories_present") or []
        family_summary = card.get("family_evidence_summary") or {}
        evidence_strength_summary = card.get("evidence_strength_summary") or "None"
        evidence_strength_tier = card.get("evidence_strength_tier") or "None"
        context_strength_tier = card.get("context_strength_tier") or "None"
        overall_priority_tier = card.get("overall_priority_tier") or "None"
        block = [
            f"Candidate {index}: {card.get('name', 'UNKNOWN')}",
            f"miRNA: {card.get('mirna_name', '')}",
            f"Gene: {card.get('gene_symbol', '')}",
            f"Support count: {card.get('support_count', 0)}",
            f"Overall priority: {overall_priority_tier}",
            f"Evidence support count: {card.get('evidence_family_count', card.get('evidence_support_count', card.get('number_of_features_supporting_interaction', 0)))} families",
            "Evidence families: " + ("; ".join(evidence_categories_present) if evidence_categories_present else "None"),
            "Overall evidence support percentile: "
            + (
                f"{float(card.get('overall_evidence_support_percentile')):.1f}"
                if card.get("overall_evidence_support_percentile") is not None and not pd.isna(card.get("overall_evidence_support_percentile"))
                else "Not available"
            ),
            f"Evidence strength tier: {evidence_strength_tier}",
            f"Context strength tier: {context_strength_tier}",
            f"Evidence strength summary: {evidence_strength_summary}",
            "Pathways: " + ("; ".join(pathway_names) if pathway_names else "Not pathway-filtered"),
            "Predicted miRNA effect on target: " + str(card.get("predicted_mirna_effect_on_target") or "unknown"),
            "Expected target role: " + str(card.get("expected_target_effect_on_phenotype") or "unknown"),
            "Target role evidence: " + (
                "; ".join(card.get("target_role_evidence") or [])
                if card.get("target_role_evidence") else "Absent"
            ),
            "Positive-regulator evidence: " + (
                "; ".join(card.get("positive_regulator_evidence") or [])
                if card.get("positive_regulator_evidence") else "Absent"
            ),
            "Negative-regulator evidence: " + (
                "; ".join(card.get("negative_regulator_evidence") or [])
                if card.get("negative_regulator_evidence") else "Absent"
            ),
            "Directional consistency: " + str(card.get("target_role_evidence_status") or "unknown"),
            "Other evidence details: " + ("; ".join(card.get("strongest_features") or []) or "None"),
            "Raw key values: " + raw_value_line,
            "Caveats: " + ("; ".join(card.get("caveats") or []) or "None"),
        ]
        for family_name in [
            "sequence_complementarity",
            "thermodynamic_stability",
            "sequence_conservation",
            "target_site_accessibility",
            "functional_binding",
            "functional_repression",
        ]:
            info = family_summary.get(family_name) or {}
            key_evidence = info.get("key_evidence") or []
            support_percentile = info.get("support_percentile")
            block.append(
                f"{family_name}: "
                + (
                    f"available={bool(info.get('available'))}; "
                    + (
                        f"support_percentile={float(support_percentile):.1f}; "
                        if support_percentile is not None and not pd.isna(support_percentile)
                        else "support_percentile=Not available; "
                    )
                    + ("key_evidence=" + "; ".join(key_evidence) if key_evidence else "key_evidence=None")
                )
            )
        block.append("Pathway evidence: " + (card.get("primary_pathway_evidence") or "Not pathway-filtered"))
        if card.get("notes"):
            block.append(f"Notes: {card.get('notes')}")
        card_blocks.append("\n".join(block))

    return instr + "\n\n" + "\n\n".join(card_blocks)


def build_prompt_bundle(
    *,
    queryspec: Optional[Dict[str, Any]] = None,
    shortlist: Optional[pd.DataFrame] = None,
    direction: Optional[str] = None,
    user_question: Optional[str] = None,
    cancer: Optional[str] = None,
    cancer_name: Optional[str] = None,
    novel: bool = False,
    phenotype_keywords: Optional[List[str]] = None,
    pathway_keywords: Optional[List[str]] = None,
    needs_clarification: Optional[List[str]] = None,
    cards: Optional[List[Dict[str, Any]]] = None,
    df: Optional[pd.DataFrame] = None,
    cards_from_dataframe: Optional[Callable[..., List[Dict[str, Any]]]] = None,
    tcga: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    retrieval_diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    qs = queryspec or {}

    if shortlist is not None and df is None:
        df = shortlist

    if user_question is None:
        user_question = qs.get("original_question") or qs.get("question") or ""

    if direction is None:
        mode = qs.get("mode")
        if mode == "mirna_to_targets":
            direction = "mirna_to_targets"
        elif mode == "gene_to_mirnas":
            direction = "gene_to_mirnas"
        else:
            direction = qs.get("direction") or "unknown"

    qs_cancer = qs.get("cancer") or {}
    if tcga is None:
        tcga = qs_cancer.get("tcga") or qs.get("tcga")
    if cancer_name is None:
        cancer_name = qs_cancer.get("name")
    if cancer is None:
        cancer = cancer_name or tcga

    if qs.get("novel") is True:
        novel = True

    if phenotype_keywords is None:
        phenotype_keywords = qs.get("phenotype_keywords") or []
    if pathway_keywords is None:
        pathway_keywords = qs.get("pathway_keywords") or []
    if needs_clarification is None:
        needs_clarification = qs.get("needs_clarification") or []
    pathway_selection = qs.get("pathway_selection") or {}
    phenotype_context = qs.get("phenotype_context") or {}
    target_role_inference = qs.get("target_role_inference") or {}
    requested_results = qs.get("result_count")
    if requested_results in (None, "", 0):
        requested_results = get_default_result_count()

    if cards is None:
        if df is None:
            raise ValueError("build_prompt_bundle requires `shortlist` (or `df`) unless `cards` are provided.")
        if cards_from_dataframe is None:
            from backend.cards import cards_from_dataframe as _cards_from_dataframe

            cards_from_dataframe = _cards_from_dataframe
        cards = cards_from_dataframe(df, tcga=tcga)

    user_prompt = build_user_prompt(
        user_question=user_question,
        direction=direction,
        cancer=tcga,
        cancer_name=cancer_name,
        novel=bool(novel),
        phenotype_keywords=phenotype_keywords or [],
        pathway_keywords=pathway_keywords or [],
        pathway_selection=pathway_selection,
        phenotype_context=phenotype_context,
        target_role_inference=target_role_inference,
        cards=cards,
        needs_clarification=needs_clarification or [],
        requested_results=int(requested_results) if requested_results is not None else None,
        retrieval_diagnostics=retrieval_diagnostics or {},
    )

    bundle: Dict[str, Any] = {
        "system_prompt": get_system_prompt(),
        "user_prompt": user_prompt,
    }

    if meta is not None:
        bundle["meta"] = meta
    elif queryspec is not None:
        bundle["meta"] = {"queryspec": queryspec}

    bundle.setdefault("meta", {})
    bundle["meta"]["cards_count"] = len(cards or [])
    bundle["meta"]["candidate_order_sent_to_llm"] = [
        {
            "rank": index,
            "gene_symbol": card.get("gene_symbol"),
            "mirna_name": card.get("mirna_name"),
            "mirassist_xgboost_score": (card.get("raw_key_values") or {}).get("mirassist_xgboost_score"),
            "score_column_used": (card.get("raw_key_values") or {}).get("score_column_used"),
            "evidence_family_count": card.get("evidence_family_count"),
            "evidence_families_present": card.get("evidence_families_present") or card.get("evidence_categories_present"),
            "raw_key_values": card.get("raw_key_values") or {},
        }
        for index, card in enumerate(cards or [], start=1)
    ]
    if retrieval_diagnostics is not None:
        bundle["meta"]["retrieval_diagnostics"] = retrieval_diagnostics

    return bundle
