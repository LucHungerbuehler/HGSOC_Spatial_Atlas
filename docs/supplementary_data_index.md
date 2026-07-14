# Supplementary Data Index

Small derived supplementary tables referenced in the thesis supplement are
included in `supplementary_data/`. The original local source paths and checksums
are recorded in `supplementary_data_manifest.csv`.

| Supplement item | GitHub-relative file |
| --- | --- |
| Supplementary Table S1: full bulk RNA-seq differential-expression workbook | `supplementary_data/signature/tt_PEO4-SNAI1-2R_Analysis.xlsx` |
| Supplementary Table S2: ORA input DEG list | `supplementary_data/signature/ora_deg_lists.csv` |
| Supplementary Table S3: ORA Hallmark, GO, and KEGG results | `supplementary_data/signature/ora_bulk_degs_all.csv` |
| Supplementary Data File: SNAI1-ac spatial robustness | `supplementary_data/signature_robustness/snai1ac_spatial_robustness_data.xlsx` |
| Supplementary Table S4: Hallmark pathway correlation meta-analysis | `supplementary_data/hallmark_composition/s2a_meta_analysis.csv` |
| Supplementary Table S5: per-sample Hallmark pathway correlations | `supplementary_data/hallmark_composition/s2a_per_sample_correlations.csv` |
| Supplementary Table S6: pooled cell-type correlations | `supplementary_data/hallmark_composition/celltype_correlations_summary.csv` |
| Supplementary Table S7: SpaGCN k=5 domain alignment summary | `supplementary_data/spagcn_lisa/domain_alignment_spagcn5.csv` |
| Supplementary Table S8: SpaGCN k=5 exploratory domain annotation summary | `supplementary_data/spagcn_lisa/spagcn5_domain_annotation_distribution_summary.csv` |
| Supplementary Table S9: C-SIDE Stouffer meta ranking, epithelial | `supplementary_data/cside/meta_ranking_Epithelial.csv` |
| Supplementary Table S9: C-SIDE Stouffer meta ranking, fibroblast | `supplementary_data/cside/meta_ranking_Fibroblast.csv` |
| Supplementary Table S9: C-SIDE Stouffer meta ranking, macrophage | `supplementary_data/cside/meta_ranking_Macrophage.csv` |
| Supplementary Table S9: C-SIDE Stouffer meta ranking, CAF | `supplementary_data/cside/meta_ranking_CAF.csv` |
| Supplementary Table S9: C-SIDE Stouffer meta ranking, endothelial | `supplementary_data/cside/meta_ranking_Endothelial.csv` |
| Supplementary Table S11: K* programme annotation evidence workbook | `supplementary_data/misc/kstar_program_annotation_evidence_supplement_v0_1.xlsx` |
| Supplementary Table S12: compact K* programme annotation workbook | `supplementary_data/misc/kstar_program_annotations_v0_2_authoritative.xlsx` |
| Supplementary Table S13: local programme to metaprogram assignment | `supplementary_data/cnmf/manual_subcluster_program_members.csv` |
| Supplementary Table S14: locked metaprogram signature genes | `supplementary_data/cnmf/MP_signatures_locked_long.csv` |
| Supplementary Table S15: metaprogram ORA results with GO audit | `supplementary_data/cnmf/MP_signature_ORA_long_with_GO_audit.csv` |
| Supplementary Table S16: metaprogram label and recurrent-gene summary | `supplementary_data/cnmf/MP_cluster_summary_labelled.csv` |
| Supplementary Table S17: GASTON gradient identity evidence | `supplementary_data/gaston/gradient_identity_evidence.csv` |
| Supplementary Data File: GASTON score--isodepth correlations | `supplementary_data/gaston/cohort_relationship_summary.csv` |
| Supplementary Data File: GASTON domain-wise fitted-score diagnostics | `supplementary_data/gaston/domainwise_score_fit_summary.csv` |
| Supplementary Data File: GASTON patient-level gene-gradient summary | `supplementary_data/gaston/gene_gradient_sample_summary.csv` |
| Supplementary Data File: GASTON signature gradient-class summary | `supplementary_data/gaston/signature_gradient_class_summary.csv` |

The GASTON tables were rebuilt from the promoted `GASTON_method_v1` branch with
`scripts/python/build_gaston_supplementary_data.py`. The JSON manifest in the
same directory records row counts and checksums for the exported tables.
