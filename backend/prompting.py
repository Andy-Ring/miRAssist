"""
Prompt templates + bundle builder for miRAssist.

Your backend imports:
    from backend.prompting import build_prompt_bundle

So this file MUST provide:
- SYSTEM_PROMPT
- build_user_prompt(...)
- build_prompt_bundle(...)

Design goals:
- Clean, uniform output
- No duplicate candidates
- Accurate method referencing
- Less skeptical tone (still evidence-grounded)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable

import pandas as pd

import json

# ----------------------------
# System prompt (synthesizer)
# ----------------------------
SYSTEM_PROMPT = """You are miRAssist, a scientific assistant that helps prioritize miRNA–mRNA interactions for experimental follow-up.

Your primary goal is to present results clearly, consistently, and accurately, using only the provided evidence.

Hard rules (must follow):
- Use ONLY the provided evidence cards and the user’s question/context. Do not invent evidence.
- If the user asks for "top N", output EXACTLY N UNIQUE ranked items.
- Never list the same gene or miRNA more than once, even if multiple cards reference it.
- Do not repeat the same candidate under different wording.
- Do not speculate beyond what is supported by the evidence cards.

Evidence interpretation rules:
- miRTarBase (functional=1): curated experimental validation (strongest support).
- ENCORI CLIP: physical binding evidence (AGO/RBP-supported), not necessarily repression.
- TargetScan context++: Sequence complementarity-based prediction (more negative = stronger).
- miRDB score: expression-based prediction (higher = stronger). A score over 60 is considered evidence for coregulation and over 80 is strong evidence.
- TCGA correlation, an anticorrelation is evidence of functional repression in a specific cancer type but lack of anticorrelation is not necessarily evidence of no interaction.

Novel mode rules:
- In novel mode, miRTarBase functional pairs must NOT appear in the ranked list.
- Known interactions may be mentioned only as background context, not as novel candidates.

Required output structure (always follow this format):

1) Interpretation
   - 1–2 sentences restating the miRNA/gene, cancer context, and phenotype/pathway (if provided).

2) Ranked Results
   - EXACTLY N items.
   - Use Markdown to create clean, easily readable bullet points.
   - Each item must be UNIQUE.
   - For each item include:
     • Name (gene symbol or miRNA)
     • Evidence summary (one compact line; only sources present in cards)
     • Context relevance (one short sentence, grounded in evidence)

3) Final Recommendation
   - 3-5 sentance summary of your final recommendation that answers the question: Which candidate would you test and why, using the evidence you just found?

Keep language professional, direct, and non-redundant.
Avoid excessive caveats or hedging.
"""


# ----------------------------
# User prompt builder
# ----------------------------
def build_user_prompt(
    *,
    user_question: str,
    direction: str,
    cancer: Optional[str] = None,
    cancer_name: Optional[str] = None,
    novel: bool = False,
    phenotype_keywords: Optional[List[str]] = None,
    pathway_keywords: Optional[List[str]] = None,
    cards: List[Dict[str, Any]],
    needs_clarification: Optional[List[str]] = None,
) -> str:
    phenotype_keywords = phenotype_keywords or []
    pathway_keywords = pathway_keywords or []
    needs_clarification = needs_clarification or []

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
        c = cancer_name if cancer_name else cancer
        ctx_lines.append(f"- Cancer context: {c}")

    if phenotype_keywords:
        ctx_lines.append(f"- Phenotype keywords: {', '.join(phenotype_keywords)}")

    if pathway_keywords:
        ctx_lines.append(f"- Pathway keywords: {', '.join(pathway_keywords)}")

    if novel:
        ctx_lines.append("- Mode: NOVEL (exclude miRTarBase functional interactions from ranked list)")

    if needs_clarification:
        ctx_lines.append(f"- Ambiguities noted by planner: {', '.join(needs_clarification)}")

    instr = f"""Task:
{task}

Requirements:
- If the user asks for "top N", provide EXACTLY N UNIQUE ranked {output_item}. Do not repeat any item.
- Use only the evidence cards below; do not invent extra support.
- Rank using strength + consistency of evidence across sources.
- For each ranked item, give a summary of the evidence for and against an interaction.
- Do not treat computational predictions as experimental proof.
- Keep phenotype/pathway discussion brief; use it mainly as a tie-breaker if evidence supports it.
- Use the required output structure from the system prompt.

User question:
{user_question.strip()}

Context:
{chr(10).join(ctx_lines) if ctx_lines else "- (none provided)"}

Evidence cards:
"""

    card_blocks: List[str] = []
    for c in cards:
        name = c.get("name", "UNKNOWN")
        evidence = c.get("evidence", "")
        notes = c.get("notes", None)

        block = [f"Candidate: {name}", f"Evidence: {evidence}"]
        if notes:
            block.append(f"Notes: {notes}")
        card_blocks.append("\n".join(block))

    return instr + "\n\n" + "\n\n".join(card_blocks)


# ----------------------------
# Bundle builder (what app.py imports)
# ----------------------------
def build_prompt_bundle(
    *,
    # ✅ what your backend/app.py passes:
    queryspec: Optional[Dict[str, Any]] = None,
    shortlist: Optional[pd.DataFrame] = None,  # alias for df
    direction: Optional[str] = None,

    # Backward-compatible:
    user_question: Optional[str] = None,
    cancer: Optional[str] = None,
    cancer_name: Optional[str] = None,
    novel: bool = False,
    phenotype_keywords: Optional[List[str]] = None,
    pathway_keywords: Optional[List[str]] = None,
    needs_clarification: Optional[List[str]] = None,

    # Cards path:
    cards: Optional[List[Dict[str, Any]]] = None,

    # If someone calls with df instead of shortlist:
    df: Optional[pd.DataFrame] = None,

    # Optional override for card builder:
    cards_from_dataframe: Optional[Callable[..., List[Dict[str, Any]]]] = None,

    # Optional:
    tcga: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a prompt bundle consumed by your synthesizer.

    - Supports your backend call: build_prompt_bundle(queryspec=..., shortlist=..., direction=...)
    - Also supports older patterns: df=..., cards=..., etc.
    """

    qs = queryspec or {}

    # Resolve DF input
    if shortlist is not None and df is None:
        df = shortlist

    # Question
    if user_question is None:
        user_question = qs.get("original_question") or qs.get("question") or ""

    # Direction
    if direction is None:
        mode = qs.get("mode")
        if mode == "mirna_to_targets":
            direction = "mirna_to_targets"
        elif mode == "gene_to_mirnas":
            direction = "gene_to_mirnas"
        else:
            direction = qs.get("direction") or "unknown"

    # Cancer context (QuerySpec uses nested cancer struct)
    qs_cancer = qs.get("cancer") or {}
    if tcga is None:
        tcga = qs_cancer.get("tcga") or qs.get("tcga")
    if cancer_name is None:
        cancer_name = qs_cancer.get("name")
    if cancer is None:
        cancer = cancer_name or tcga

    # Novel
    if qs.get("novel") is True:
        novel = True

    # Keywords / clarifications
    if phenotype_keywords is None:
        phenotype_keywords = qs.get("phenotype_keywords") or []
    if pathway_keywords is None:
        pathway_keywords = qs.get("pathway_keywords") or []
    if needs_clarification is None:
        needs_clarification = qs.get("needs_clarification") or []

    # Build cards
    if cards is None:
        if df is None:
            raise ValueError("build_prompt_bundle requires `shortlist` (or `df`) unless `cards` are provided.")

        if cards_from_dataframe is None:
            # Default import to match your project layout
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
        cards=cards,
        needs_clarification=needs_clarification or [],
    )

    bundle: Dict[str, Any] = {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
    }

    if meta is not None:
        bundle["meta"] = meta
    elif queryspec is not None:
        bundle["meta"] = {"queryspec": queryspec}

    return bundle
