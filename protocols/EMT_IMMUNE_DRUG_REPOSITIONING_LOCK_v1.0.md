# EMT–immune state drug-repositioning and combination-discovery lock v1.0

## Scientific objective

Identify compounds that can reverse state-specific EMT–immune bottlenecks in breast cancer and rank rational two-agent combinations that act on complementary compartments or immune-cycle defects. This is an outcome-free mechanism and perturbation study, not a treatment-response prediction analysis.

## Biological states

1. **Epithelial cold / initiation-poor:** weak antigen visibility and T/NK recruitment.
2. **Hybrid immune-visible barrier:** simultaneous epithelial–mesenchymal coactivation, immune visibility/recruitment, and TGF-beta/ECM or stromal-barrier activity.
3. **Mesenchymal engaged–incomplete:** immune contact/visibility with reduced effective execution and barrier/checkpoint pressure.

Double-low is retained as an indeterminate reference and is not assigned a drug hypothesis.

Drug nomination is state-specific. No universal compound or combination is assumed. A shared agent may appear in more than one state only when the state-specific pathway evidence independently supports it; its partner and mechanistic purpose may differ by state.

## Pathway-first analysis

Pathway analysis precedes compound ranking.

- Hallmark provides compact, low-redundancy biological themes.
- Reactome provides mechanistic pathway resolution and hierarchy.
- GO/KEGG may be used as secondary confirmation when locally available, but redundant database hits are collapsed rather than counted as independent evidence.
- OmniPath is used after pathway discovery to resolve directed signalling and ligand–receptor/compartment relationships.
- Pathway activity is analysed continuously against within-patient EMT position and E/M coactivation with patient-clustered inference, and descriptively/inferentially in paired state contrasts.
- Pairwise malignant-state contrasts are E versus hybrid, hybrid versus mesenchymal, and E versus mesenchymal. Contrasts with fewer than six paired patients are explicitly low-information and cannot define a drug query alone.
- Cell-cycle and stress axes are adjusted in continuous malignant-metacell models.
- Pathway FDR is controlled separately by database and EMT term/contrast.

## Evidence layers and compartment rule

- Malignant-cell transcriptomics define tumour-intrinsic reversal signatures.
- Bulk tumours define the whole-ecosystem state and provide independent replication, not a substitute for cellular origin.
- Single-cell and spatial data determine whether a target belongs to malignant, immune or stromal compartments.
- A cancer-cell LINCS perturbation cannot by itself validate a stromal or immune mechanism. Such mechanisms require target-expression and external perturbation evidence in the relevant compartment.

## Stage 1 — malignant-cell candidate signatures

Source: outcome-free GSE176078 malignant metacells and their previously frozen EMT-state assignments.

- Aggregate metacell counts within patient and state.
- For each target state, compare its patient pseudobulk with the same patient's pooled pseudobulk from the other two interpretable EMT states.
- Double-low is excluded from the primary comparator.
- Normalize as log2(CPM + 0.5).
- Report paired mean log2 fold-change, median fold-change, paired t statistic, nominal P value, BH FDR, and patient-level direction consistency.
- Candidate query eligibility: mean CPM >= 1, absolute mean paired log2FC >= 0.25, direction consistency >= 2/3, and representation among LINCS best-inferred genes.
- Within each state and direction, rank by absolute paired t statistic, then absolute effect; retain at most 100 up and 100 down genes. At least 25 genes per direction are required.
- These are discovery signatures and cannot be interpreted as confirmed state markers until independent replication.

The whole-state signatures are diagnostic discovery outputs, not the primary drug query. A drug should repair the state-specific bottleneck rather than indiscriminately erase every feature of the state. In particular, immune visibility in hybrid or mesenchymal tumours is a feature to preserve, while TGF-beta/ECM or checkpoint/barrier pressure is a feature to suppress.

## Primary bottleneck-repair objectives

- **Epithelial cold:** induce tumour-cell IFN response, antigen processing/presentation and T/NK-recruiting chemokines; penalize simultaneous induction of TGF-beta/ECM barrier.
- **Hybrid visible-barrier:** suppress tumour-cell TGF-beta/ECM barrier while preserving antigen visibility, IFN response and T/NK recruitment.
- **Mesenchymal engaged–incomplete:** suppress tumour-cell barrier/checkpoint-ligand pressure while preserving visibility; restoration of T/NK execution is evaluated as an immune-compartment partner objective, not inferred from cancer-cell LINCS alone.

Accordingly, compound scoring is multi-objective. Whole-state negative connectivity is supportive only and cannot override a failure to satisfy the bottleneck-repair objective.

## Stage 2 — independent signature replication

- Replicate gene direction in TCGA-BRCA with PAM50 and stromal adjustment.
- Seek concordance in METABRIC and SCAN-B where platform coverage permits.
- Separate tumour-intrinsic genes from ecosystem genes using GSE176078 and spatial compartment evidence.
- Freeze the replicated query before reading compound connectivity scores.

## Stage 3 — perturbational repositioning

- Query LINCS L1000 Level 5 signatures using a signed rank-based connectivity statistic.
- Primary cell context: breast cancer lines; pan-cancer evidence is supportive.
- Aggregate replicate signatures by compound, cell line, dose and time without selecting on desired direction.
- Prioritize negative connectivity (reversal), consistency across breast models, perturbation quality, and clinically interpretable targets.
- Generic cytotoxicity, low-quality signatures and single-context hits are penalized.

## Stage 4 — orthogonal pharmacogenomic support

- PRISM and GDSC test whether state-matched breast models are selectively sensitive to nominated compounds.
- CCLE expression defines EMT state in cell lines without using drug response.
- Sensitivity association and LINCS reversal are independent evidence streams; neither is required to be significant merely because the other is favorable.

## Stage 5 — combination ranking

A pair is ranked only when the agents address complementary bottlenecks or compartments. The score integrates:

- replicated state-signature reversal;
- target expression in the correct compartment;
- breast-model sensitivity or dependency support;
- mechanistic complementarity;
- existing combination/synergy evidence when available;
- clinical development status and non-overlapping toxicity concerns.

No computational score establishes clinical efficacy or synergy. Final claims are mechanistic hypotheses for experimental validation.

## Multiplicity and anti-overfitting safeguards

- Treatment outcomes are not used for signature creation, drug ranking or pair construction.
- No gene, compound, target or threshold may be changed because it improves a desired drug rank.
- Discovery, replication, perturbation and pharmacogenomic evidence are reported separately before integration.
- Literature is used to annotate and falsify ranked hypotheses, not to retrofit the signatures.
