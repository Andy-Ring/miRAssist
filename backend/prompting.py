"""
Prompt templates + bundle builder for miRAssist.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd


SYSTEM_PROMPT = """You are miRAssist, a scientific assistant that helps prioritize miRNA-mRNA interactions for experimental follow-up.

Your primary goal is to present results clearly, consistently, and accurately, using only the provided evidence.

Hard rules:
- Use ONLY the provided evidence cards and the user's question/context. Do not invent evidence.
- The backend has already ranked the candidates. Use the provided ranking order and do not rerank based on unsupported intuition.
- Output EXACTLY the requested number of ranked items when enough candidates are available.
- Never list the same gene or miRNA more than once.
- Do not invent gene-to-phenotype links.
- Do not invent pathway membership.
- Do not infer pathway directionality unless the pathway name explicitly supports it.
- Do not calculate percentiles yourself; use the backend-provided labels and values.
- Do not invent feature strengths; use only the backend-provided raw values and labels.
- Use the backend-provided evidence_support_count exactly as given.
- Percentiles are interpretation labels for the associated feature, not separate evidence.
- Do not count a raw feature and its percentile annotation separately.
- Evidence support count is based on distinct evidence categories, not individual columns.
- Use at most one primary metric per evidence category unless the card explicitly marks additional metrics as nonredundant.
- Do not list best/mean/max variants of the same evidence type as separate key evidence unless the card makes that necessary.
- Do not include low or absent evidence as a key evidence line unless it is an important caveat.
- Do not describe computational predictions as experimental validation.

Evidence interpretation rules:
- miRTarBase functional support is curated prior evidence, not automatic proof of the exact user context.
- miRDB, TargetScan, seed features, RNAhybrid, and local AU are computational or structural support, not experimental validation.
- ENCORI/CLIP supports binding evidence, not necessarily repression.
- TCGA context evidence is context-specific repression support, not direct binding evidence.
- Pathway evidence only means the candidate passed the deterministic pathway filter or has explicit pathway names in the card.

Novel mode rules:
- In novel mode, miRTarBase functional pairs must NOT appear in the ranked list.
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
- Return candidates in the provided ranking order.
- If fewer than the requested number of candidates were provided, say that fewer candidates passed the filters.
- For each result use this format:

### 1. GENE_OR_MIRNA
- **Evidence support count:** X categories
- **Evidence categories:** miRTarBase; miRDB; TargetScan; CLIP; TCGA context; Pathway
- **Pathways:** pathway names if pathway filtering was applied, otherwise "Not pathway-filtered"
- **Key pieces of evidence:**
  - grouped by evidence category, with one primary metric per category and percentiles treated only as modifiers
- **Explanation:** 2-4 sentences grounded only in the evidence card

## Final recommendation
- A short paragraph saying which 1-3 candidates are best for follow-up and why.

Keep language professional, direct, and readable.
"""


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
    cards: List[Dict[str, Any]],
    needs_clarification: Optional[List[str]] = None,
    requested_results: Optional[int] = None,
    retrieval_diagnostics: Optional[Dict[str, Any]] = None,
) -> str:
    phenotype_keywords = phenotype_keywords or []
    pathway_keywords = pathway_keywords or []
    pathway_selection = pathway_selection or {}
    needs_clarification = needs_clarification or []
    retrieval_diagnostics = retrieval_diagnostics or {}

    if direction == "mirna_to_targets":
        task = "Identify and rank target genes regulated by the miRNA."
        output_item = "genes"
    elif direction == "gene_to_mirnas":
        task = "Identify and rank miRNAs that regulate the gene."
        output_item = "miRNAs"
    else:
        task = "Identify and rank relevant candidates from the evidence."
        output_item = "candidates"

    ctx_lines: List[str] = []
    if cancer_name or cancer:
        ctx_lines.append(f"- Cancer context: {cancer_name if cancer_name else cancer}")
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
    if novel:
        ctx_lines.append("- Mode: NOVEL (exclude miRTarBase functional interactions from ranked list)")
    if needs_clarification:
        ctx_lines.append(f"- Ambiguities noted by planner: {', '.join(needs_clarification)}")

    available_n = len(cards)
    top_n = int(requested_results or available_n)
    if top_n <= 0:
        top_n = available_n
    instr = f"""Task:
{task}

Requirements:
- Requested ranked results: {top_n}
- Available evidence cards: {available_n}
- Return EXACTLY {top_n} UNIQUE ranked {output_item} in the provided order unless fewer than {top_n} candidates passed the filters.
- Use only the evidence cards below; do not invent extra support.
- Do not rerank candidates based on unsupported intuition.
- Use the backend-provided evidence support count, evidence categories, raw values, percentile labels, and caveats.
- Use the backend-provided evidence_support_count exactly as given.
- Percentiles are interpretation labels for the associated feature, not separate evidence.
- Do not count raw values and percentiles separately.
- Use at most one primary metric per evidence category unless the card explicitly marks additional metrics as nonredundant.
- Keep "Key pieces of evidence" compact and grouped by category.
- Do not invent percentiles, feature strength labels, pathway membership, or gene-phenotype links.
- Do not claim a candidate belongs to a pathway unless the card explicitly says it passed the pathway filter or names the pathways.
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
        evidence_categories_present = card.get("evidence_categories_present") or []
        primary_curated = card.get("primary_curated_evidence")
        primary_mirdb = card.get("primary_mirdb_evidence")
        primary_targetscan = card.get("primary_targetscan_evidence")
        primary_clip = card.get("primary_clip_evidence")
        primary_seed = card.get("primary_seed_evidence")
        primary_structure = card.get("primary_structure_evidence")
        primary_tcga = card.get("primary_tcga_evidence")
        primary_pathway = card.get("primary_pathway_evidence")
        block = [
            f"Candidate {index}: {card.get('name', 'UNKNOWN')}",
            f"miRNA: {card.get('mirna_name', '')}",
            f"Gene: {card.get('gene_symbol', '')}",
            f"Support count: {card.get('support_count', 0)}",
            f"Evidence support count: {card.get('evidence_support_count', card.get('number_of_features_supporting_interaction', 0))} categories",
            "Evidence categories: " + ("; ".join(evidence_categories_present) if evidence_categories_present else "None"),
            "Pathways: " + ("; ".join(pathway_names) if pathway_names else "Not pathway-filtered"),
            "Curated evidence: " + (primary_curated or "None"),
            "miRDB evidence: " + (primary_mirdb or "None"),
            "TargetScan evidence: " + (primary_targetscan or "None"),
            "Binding evidence: " + (primary_clip or "None"),
            "Seed/site evidence: " + (primary_seed or "None"),
            "Structure evidence: " + (primary_structure or "None"),
            "Context evidence: " + (primary_tcga or "None"),
            "Pathway evidence: " + (primary_pathway or "Not pathway-filtered"),
            "Other evidence details: " + ("; ".join(card.get("strongest_features") or []) or "None"),
            "Raw key values: " + raw_value_line,
            "Caveats: " + ("; ".join(card.get("caveats") or []) or "None"),
        ]
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
    requested_results = qs.get("k") or (len(df) if df is not None else len(cards or []))

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
        cards=cards,
        needs_clarification=needs_clarification or [],
        requested_results=int(requested_results) if requested_results is not None else None,
        retrieval_diagnostics=retrieval_diagnostics or {},
    )

    bundle: Dict[str, Any] = {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
    }

    if meta is not None:
        bundle["meta"] = meta
    elif queryspec is not None:
        bundle["meta"] = {"queryspec": queryspec}

    bundle.setdefault("meta", {})
    bundle["meta"]["cards_count"] = len(cards or [])
    if retrieval_diagnostics is not None:
        bundle["meta"]["retrieval_diagnostics"] = retrieval_diagnostics

    return bundle
