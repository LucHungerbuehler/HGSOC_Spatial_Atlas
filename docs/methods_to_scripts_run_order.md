# Methods-to-Scripts Run Order

This document maps the thesis Methods chapter to the staged analysis scripts.
The package is GitHub-first and does not include raw/public source datasets or large derived objects.

## Spatial preprocessing

- `scripts/python/preprocessing.py` (core, available)
- `scripts/python/organize_visium.py` (support, available)
- `scripts/python/run_qc_pipeline.py` (support, available)

## Bulk signature derivation

- `scripts/python/Signature_Weights_prep.py` (core, available)
- `scripts/R/ora_bulk_DEGs.R` (core, available)
- `scripts/R/ora_heatmaps.R` (support, available)

## Spatial gene-set scoring

- `scripts/python/enrichmap_scoring.py` (core, available)
- `scripts/python/signature_robustness_audit.py` (sensitivity, available)

## Pathway/composition co-localization

- `scripts/python/hallmark_correlation_analysis.py` (core, available)
- `scripts/python/tumor_only_hallmark_analysis.py` (core, available)
- `scripts/python/celltype_composition_colocalization.py` (core, available)
- `scripts/python/ilr_composition_analysis.py` (core, available)
- `scripts/python/plot_s2b_ilr_predictability_figures.py` (figure, available)
- `scripts/python/build_s2b_step4_threshold_diagnostics.py` (sensitivity, available)
- `scripts/python/build_s2b_step4_threshold_diagnostics_10_to_100.py` (sensitivity, available)
- `scripts/python/plot_s2b_step4_ilr_group_positions.py` (support, available)
- `scripts/python/plot_s2b_step4_hallmark_group_positions.py` (support, available)

## SpaGCN/LISA

- `scripts/python/spatial_signature_analysis.py` (core, available)
- `scripts/python/spagcn_matched_malignancy_domain_analysis.py` (core, available)
- `scripts/python/plot_sign_figs.py` (support, available)

## C-SIDE

- `scripts/R/run_rctd_and_cside.r` (core, available)
- `scripts/python/build_cside_section25_assets.py` (support, available)
- `scripts/python/build_cside_report_ready_outputs.py` (support, available)
- `scripts/python/cside_per_gene_direction_sign_test.py` (sensitivity, available)
- `scripts/python/cside_setlevel_reproducibility_permutation.py` (sensitivity, available)
- `scripts/python/cside_audit_sensitivity.py` (sensitivity, available)
- `scripts/python/cside_model1_source_integrity_audit.py` (audit, available)
- `scripts/python/cside_model1_vs_model2_compact_audit.py` (audit, available)
- `scripts/python/cside_model1_vs_model2_deep_gene_audit.py` (audit, available)
- `scripts/R/run_cside_qc_sensitivity.r` (sensitivity, available)

## Tumor cNMF

- `scripts/python/s3_tumor_cnmf_pipeline.py` (core, available)
- `scripts/python/jaccard_metaprogram_pipeline.py` (core, available)
- `scripts/python/score_manual_jaccard_subclusters_enrichmap.py` (core, available)
- `scripts/python/regenerate_coarse_recluster_heatmaps_no_fine.py` (support, available)
- `scripts/python/snai1ac_program_decomposition_v1.py` (core, available)
- `scripts/python/snai1ac_program_decomposition_unsmoothed_uncorrected_v1.py` (sensitivity, available)
- `scripts/python/compute_cnmf_usage_norm_only_model.py` (sensitivity, available)
- `scripts/python/build_cnmf_model_fit_report_examples.py` (figure, available)
- `scripts/python/plot_cnmf_cohort_cv_r2_model_comparison.py` (figure, available)
- `scripts/python/plot_snai1ac_kstar_univariate_and_model_diagnostics.py` (figure, available)
- `scripts/python/build_cnmf_univariate_high_corr_table.py` (figure, available)
- `scripts/python/snai1ac_mp_three_arm_correlation.py` (core, available)
- `scripts/R/snai1ac_mp_meta_analysis.R` (core, available)
- `scripts/python/snai1ac_mp_lisa_hotspot_contrasts.py` (core, available)
- `scripts/R/snai1ac_mp_lisa_meta_analysis.R` (core, available)
- `scripts/python/snai1ac_mp_existing_scores_tumor_filter_correlation.py` (support, available)
- `scripts/R/snai1ac_mp_existing_scores_tumor_filter_meta_analysis.R` (support, available)
- `scripts/python/snai1ac_program_prediction.py` (support, available)
- `scripts/python/build_cnmf_snai1ac_signature_projection_v1.py` (support, available)
- `scripts/python/plot_cnmf_snai1ac_projection_distributions_v1.py` (support, available)
- `scripts/python/rerun_final_mp1_8_stale_analyses.py` (provenance, available)
- `scripts/python/audit_final_mp1_8_ora_with_go.py` (sensitivity, available)
- `scripts/python/analysis_utils.py` (support, available)
- `scripts/python/def34_raw_genenmf_pipeline.py` (support, available)
- `scripts/python/define_snaI1_tme_program_families.py` (support, available)
- `scripts/python/programme_state_research_synthesis.py` (support, available)
- `scripts/python/build_kstar_program_annotation_supplement.py` (support, available)
- `scripts/python/hh_programme_characterization.py` (core, available)
- `scripts/python/build_hh_composite_figure.py` (figure, available)
- `scripts/python/build_sp6_hotspot_three_contrast_regenerated.py` (figure, available)
- `scripts/python/build_hh_state_family_count_summaries.py` (figure, available)
- `scripts/python/build_hh_state_family_count_composite.py` (figure, available)
- `scripts/python/family_topgene_coherence_plots.py` (figure, available)
- `scripts/python/family_topgene_functional_order_plots.py` (figure, available)
- `scripts/python/regenerate_jaccard_manual_heatmap_no_fine.py` (figure, available)
- `scripts/python/regenerate_umap_companion_panels_reference_family.py` (figure, available)
- `scripts/python/build_hh_state_mp_enrichmap_count_composite.py` (figure, available)

## SpottedPy

- `scripts/python/spottedpy_v2_hotspot_preflight.py` (provenance, available)
- `scripts/python/spottedpy_v2_revised_hotspot_preflight.py` (core, available)
- `scripts/python/spottedpy_v2_distance_design_availability_preflight.py` (core, available)
- `scripts/python/spottedpy_v2_distance_preflight_calculate.py` (core, available)
- `scripts/python/spottedpy_v2_distance_spottedpy_native_preflight.py` (core, available)
- `scripts/python/spottedpy_v2_distance_full_promotion.py` (core, available)
- `scripts/python/spottedpy_v2_kstar_component_distance_tests_preflight.py` (support, available)
- `scripts/python/spottedpy_v2_consensus_source_neighborhood_preflight.py` (support, available)
- `scripts/python/spottedpy_v2_source_group_correlation_contrast_preflight.py` (support, available)
- `scripts/python/spottedpy_v2_neighborhood_postprocess_ranked_and_source_comparisons.py` (support, available)
- `scripts/python/spottedpy_v2_clean_allinone_snai1ac_primary.py` (support, available)
- `scripts/python/spottedpy_v2_hallmark_spacet_hotspots.py` (support, available)
- `scripts/python/spottedpy_v2_hallmark_scoregenes_neighborhood_preflight.py` (support, available)
- `scripts/python/spottedpy_v2_cross_sample_neighborhood_summaries.py` (support, available)
- `scripts/python/spottedpy_v2_refresh_consensus_neighborhood_figures.py` (figure, available)
- `scripts/python/spottedpy_v2_visual_qc_contact_sheets.py` (support, available)

## GASTON

- `scripts/python/gaston_method_aligned_preflight.py` (core, available)
- `scripts/python/gaston_method_aligned_run.py` (core, available)
- `scripts/python/gaston_build_isodepth_score_alignment.py` (core, available)
- `scripts/python/gaston_native_gene_gradient.py` (core, available)
- `scripts/python/gaston_build_gradient_identity_review.py` (support, available)
- `scripts/python/gaston_report_asset_composites.py` (support, available)

## Before Public Push

- Hard-coded local path hits found: 147.
- These paths are intentionally retained as analysis provenance for the thesis code archive.
- Do not present `scripts/python/main.py` as a complete final pipeline; it is an earlier development orchestrator.
- Keep raw data, large `.h5ad`/`.rds` objects, and rendered figure archives out of GitHub.
- Use `supplementary_data_manifest.csv` and `docs/supplementary_data_index.md` to point readers to the small derived supplementary tables included here.
