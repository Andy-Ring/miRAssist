"""
Prompt templates for miRAssist synthesis.

This file controls:
- Global model behavior and tone (SYSTEM_PROMPT)
- Task-specific user prompts (build_user_prompt)

Design goals:
- Clean, uniform output
- No duplicated candidates
- Evidence-grounded descriptions
- Minimal over-skepticism
"""

from typing import List, Dict, Optional


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
- TargetScan context++: computational binding prediction (more negative = stronger).
- miRDB score: computational prediction strength (higher = stronger).
- TCGA correlation or repression flags: supportive population-level context only (not proof of regulation).

Novel mode rules:
- In novel mode, miRTarBase functional pairs must NOT appear in the ranked list.
- Known interactions may be mentioned only as background context, not as novel candidates.

Required output structure (always follow this format):

1) Interpretation
   - 1–2 sentences restating the miRNA/gene, cancer context, and phenotype/pathway (if provided).

2) Ranked Results
   - EXACTLY N items.
   - Each item must be UNIQUE.
   - For each item include:
     • Name (gene symbol or miRNA)
     • Evidence summary (one compact line; only sources present in cards)
     • Context relevance (one short sentence, grounded in evidence)

3) Methods Note
   - 2–4 concise bullets explaining how to interpret the evidence sources used.

4) Suggested Experiments
   - 2–4 concise, realistic experimental follow-ups.

Keep language professional, direct, and non-redundant.
Avoid excessive caveats or hedging.
"""


def build_user_prompt(
    *,
    user_question: str,
    direction: str,
    cancer: Optional[str],
    cancer_name: Optional[str],
    novel: bool,
    phenotype_keywords: List[str],
    pathway_keywords: List[str],
    cards: List[Dict],
    needs_clarification: Optional[List[str]] = None,
) -> str:
    """
    Build the user-facing synthesis prompt.
    """

    if direction == "mirna_to_targets":
        task = "Identify and rank target genes regulated by the miRNA."
        output_item = "gene"
    elif direction == "gene_to_mirnas":
        task = "Identify and rank miRNAs that regulate the gene."
        output_item = "miRNA"
    else:
        task = "Identify relevant miRNA–mRNA interactions."
        output_item = "candidate"

    ctx_lines = []

    if cancer_name or cancer:
        c = cancer_name if cancer_name else cancer
        ctx_lines.append(f"- Cancer context: {c}")

    if phenotype_keywords:
        ctx_lines.append(f"- Phenotype keywords: {', '.join(phenotype_keywords)}")

    if pathway_keywords:
        ctx_lines.append(f"- Pathway keywords: {', '.join(pathway_keywords)}")

    if novel:
        ctx_lines.append("- Mode: NOVEL (exclude miRTarBase functional interactions)")

    if needs_clarification:
        ctx_lines.append(f"- Ambiguities noted by planner: {', '.join(needs_clarification)}")

    instr = f"""Task:
{task}

Requirements:
- If the user asks for "top N", provide EXACTLY N UNIQUE ranked {output_item}s.
- Do NOT repeat the same {output_item}, even if multiple evidence cards exist.
- Rank candidates using overall strength and consistency of evidence across sources.
- For each ranked {output_item}, include ONE compact evidence line using ONLY what appears in the cards:
  miRTarBase(functional=0/1), ENCORI(CLIP counts), TargetScan(best context++ and site count),
  miRDB(best score), TCGA(correlation or repression flags if present).
- Do not treat computational scores as experimental proof.
- Keep phenotype/pathway discussion brief and grounded in evidence.
- Use a clean, uniform writing style. No repetition. No filler text.

User question:
{user_question.strip()}

Context:
{chr(10).join(ctx_lines)}

Evidence cards:
"""

    card_blocks = []
    for c in cards:
        block = []
        block.append(f"Candidate: {c.get('name')}")
        block.append(f"Evidence: {c.get('evidence')}")
        if c.get("notes"):
            block.append(f"Notes: {c.get('notes')}")
        card_blocks.append("\n".join(block))

    return instr + "\n\n" + "\n\n".join(card_blocks)
