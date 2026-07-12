from __future__ import annotations

import os

import pandas as pd

os.environ.setdefault("SPOTTEDPY_V2_NEIGHBORHOOD_RUN_MODE", "full")

import spottedpy_v2_consensus_source_neighborhood_preflight as nb  # noqa: E402


def main() -> None:
    label_map = nb.load_display_label_map()
    n_heatmaps = 0
    n_ranked = 0
    n_source_comparison = 0

    matrix_path = nb.TABLE_DIR / "consensus_source_allinone_pearson_correlations.csv"
    matrix_df = pd.read_csv(matrix_path)
    for (sample_label, source_group, class_name), group in matrix_df.groupby(
        ["sample_label", "source_group", "variable_class"],
        sort=False,
    ):
        corr = group.pivot(index="variable_a", columns="variable_b", values="corr")
        variables = sorted(set(group["variable_a"]).union(group["variable_b"]))
        corr = corr.reindex(index=variables, columns=variables)
        corr = corr.combine_first(corr.T)
        for variable in variables:
            corr.loc[variable, variable] = 1.0
        nb.plot_corr_heatmap(
            corr,
            nb.FIG_DIR / f"{nb.safe_name(sample_label)}__{source_group}__{class_name}__clustered_heatmap.png",
            f"{sample_label}: {source_group}, {class_name}",
            label_map,
        )
        n_heatmaps += 1

    ranked_path = nb.TABLE_DIR / "consensus_source_central_snai1ac_ranked12_for_plots.csv"
    ranked_df = pd.read_csv(ranked_path)
    for (sample_label, source_group, class_name), group in ranked_df.groupby(
        ["sample_label", "source_group", "variable_class"],
        sort=False,
    ):
        nb.plot_ranked_bars(
            group,
            nb.FIG_DIR / f"{nb.safe_name(sample_label)}__{source_group}__{class_name}__SNAI1ac_ring1_ranked12.png",
            f"{sample_label}: {source_group}, {class_name}, SNAI1-ac to ring 1",
            label_map,
        )
        n_ranked += 1

    source_path = nb.TABLE_DIR / "consensus_source_sourcegroup_comparison_long.csv"
    source_df = pd.read_csv(source_path)
    source_order = list(nb.CORE_SOURCE_COLUMNS.keys())
    for (sample_label, class_name), group in source_df.groupby(["sample_label", "variable_class"], sort=False):
        if group["source_group"].nunique() < 2:
            continue
        nb.plot_source_group_comparison(
            group,
            nb.FIG_DIR / f"{nb.safe_name(sample_label)}__{class_name}__source_group_comparison_paperstyle.png",
            f"{sample_label}: source-group comparison, {class_name}",
            source_order,
            label_map,
        )
        n_source_comparison += 1

    print(
        {
            "output_root": str(nb.OUT_ROOT),
            "n_heatmaps": n_heatmaps,
            "n_ranked_barplots": n_ranked,
            "n_source_group_comparison_heatmaps": n_source_comparison,
        }
    )


if __name__ == "__main__":
    main()
