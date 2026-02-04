# backend/prompting.py
from __future__ import annotations

from typing import Any, Dict, List, Optional


SYSTEM_PROMPT = """You are miRAssist, a scientific assistant that helps prioritize miRNA–mRNA interactions for experimental follow-up.

Rules:
- Use ONLY the provided evidence cards and the user’s question context.
- Do not fabricate database support or statistics.
- Treat expression/correlation as supportive evidence, not proof of regulation.
- Be explicit about uncertainty and alternative hypotheses.
- If user asks for "top N", provide exactly N ranked items.
- When in NOVEL mode, do NOT recommend miRTarBase functional pairs as "novel".
"""


def build_user_prompt(
    user_question: str,
    direction: str,
    cancer: Optional[str] = None,              # TCGA code e.g., BRCA/COAD/PRAD (kept for backward compat)
    cancer_name: Optional[str] = None,         # human-readable (from planner)
    novel: bool = False,
    phenotype_keywords: Optional[List[str]] = None,
    pathway_keywords: Optional[List[str]] = None,
    cards: Optional[List[Dict[str, Any]]] = None,
    needs_clarification: Optional[List[str]] = None,
) -> str:
    """
    Create the user prompt for the synthesizer LLM.

    direction:
        "mirna_to_targets" or "gene_to_mirnas"
    """
    cards = cards or []
    phenotype_keywords = phenotype_keywords or []
    pathway_keywords = pathway_keywords or []
    needs_clarification = needs_clarification or []

    if direction == "mirna_to_targets":
        task = "Prioritize likely TARGET GENES regulated by the miRNA in the question."
        output_item = "gene"
    elif direction == "gene_to_mirnas":
        task = "Prioritize likely miRNAs that selectively REDUCE expression of the gene/mRNA in the question."
        output_item = "miRNA"
    else:
        task = "Prioritize likely miRNA–target interactions relevant to the question."
        output_item = "candidate"

    ctx_lines: List[str] = []
    if cancer_name:
        ctx_lines.append(f"- Cancer context (user): {cancer_name}")
    if cancer:
        ctx_lines.append(f"- TCGA cohort context: {cancer}")
    ctx_lines.append(f"- Mode: {'NOVEL' if novel else 'ANY'}")
    if phenotype_keywords:
        ctx_lines.append(f"- Phenotype hints: {', '.join(phenotype_keywords[:12])}")
    if pathway_keywords:
        ctx_lines.append(f"- Pathway hints: {', '.join(pathway_keywords[:12])}")

    caveat_block = ""
    if needs_clarification:
        caveat_block = (
            "\nPlanner notes / ambiguity (do NOT ask follow-ups; just caveat):\n"
            + "\n".join([f"- {x}" for x in needs_clarification[:10]])
            + "\n"
        )

    instr = f"""Task:
{task}

Requirements:
- Provide a ranked list of {output_item}s with short justifications grounded in the evidence cards. Be specific about how the evidence supports the interaction.
- Use database support signals: miRTarBase functional, ENCORI CLIP, TargetScan context++ (more negative is stronger), and miRDB scores (higher is stronger).
- If TCGA correlation/repression evidence appears in cards, use it as supportive context only.
- If phenotype/pathway hints are present, prefer {output_item}s that plausibly fit those hints *when the cards support them*.

User question:
{user_question.strip()}

Context:
{chr(10).join(ctx_lines)}
{caveat_block}
Evidence cards:
"""

    card_lines: List[str] = []
    for i, c in enumerate(cards, 1):
        card_lines.append(f"\n--- CARD {i} ---")
        card_lines.append(str(c))

    return instr + "\n".join(card_lines)


def _row_to_card(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a shortlist row to a compact evidence card.
    Keep it small and predictable to manage prompt length.
    """
    # These keys are intentionally conservative; add more later if needed.
    keep = [
        "mirna_name",
        "gene_symbol",
        "support_count",
        "support_encori",
        "support_targetscan",
        "support_mirdb",
        "mirtarbase_pos",
        "ts_best_contextpp",
        "ts_best_percentile",
        "ts_n_sites",
        "clip_exp_sum",
        "clip_exp_max",
        "n_clip_sites",
        "mirdb_best_score",
        "mirdb_mean_score",
        "retrieval_score",
    ]

    card = {k: row.get(k) for k in keep if k in row}

    # Also keep any tcga evidence columns that might exist (small subset)
    for k in list(row.keys()):
        if k.endswith("_spearman_rho") or k.endswith("_spearman_p") or k.endswith("_repression_evidence") or k.endswith("_anticorrelated"):
            card[k] = row.get(k)

    return card


def build_prompt_bundle(
    queryspec: Dict[str, Any],
    shortlist,
    max_prompt_tokens: int = 6500,
) -> Dict[str, Any]:
    """
    Build the synthesizer bundle:
      {system_prompt, user_prompt, meta}

    Note: max_prompt_tokens is carried in meta; actual truncation should happen
    upstream when selecting how many cards to include (token-aware selection).
    """
    # queryspec fields (defensive)
    question = (queryspec.get("original_question") or "").strip()
    direction = queryspec.get("mode") or queryspec.get("direction") or "mirna_to_targets"
    novel = bool(queryspec.get("novel", False))

    cancer = None
    cancer_name = None
    if isinstance(queryspec.get("cancer"), dict):
        cancer = queryspec["cancer"].get("tcga")
        cancer_name = queryspec["cancer"].get("name")
    else:
        # backward compat
        cancer = queryspec.get("tcga")
        cancer_name = queryspec.get("cancer_name")

    phenotype_keywords = queryspec.get("phenotype_keywords") or []
    pathway_keywords = queryspec.get("pathway_keywords") or []
    needs_clarification = queryspec.get("needs_clarification") or []

    # shortlist -> cards
    if hasattr(shortlist, "to_dict"):
        rows = shortlist.to_dict(orient="records")
    else:
        rows = list(shortlist)

    cards = [_row_to_card(r) for r in rows]

    user_prompt = build_user_prompt(
        user_question=question,
        direction=direction,
        cancer=cancer,
        cancer_name=cancer_name,
        novel=novel,
        phenotype_keywords=phenotype_keywords,
        pathway_keywords=pathway_keywords,
        cards=cards,
        needs_clarification=needs_clarification,
    )

    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "meta": {
            "project": "miRAssist",
            "max_prompt_tokens": int(max_prompt_tokens),
            "n_cards": len(cards),
        },
    }
