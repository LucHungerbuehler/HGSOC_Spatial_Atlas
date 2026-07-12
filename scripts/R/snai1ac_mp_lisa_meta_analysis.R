args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("Usage: Rscript snai1ac_mp_lisa_meta_analysis.R <output_dir>")

suppressPackageStartupMessages(library(metafor))

out_dir <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
tables_dir <- file.path(out_dir, "tables")
plots_dir <- file.path(out_dir, "plots", "forest")
dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

effects_path <- file.path(tables_dir, "snai1ac_lisa_per_sample_effects.csv")
if (!file.exists(effects_path)) stop("Missing effects table: ", effects_path)
d <- read.csv(effects_path, stringsAsFactors = FALSE)
need <- c("variant", "contrast", "MP_id", "MP_label", "dataset", "sample",
          "n_group1", "n_group2", "rank_biserial", "effect_for_meta",
          "included_in_meta")
missing <- setdiff(need, names(d))
if (length(missing)) stop("Effects table lacks columns: ", paste(missing, collapse = ", "))

d$included_in_meta <- as.character(d$included_in_meta) %in% c("TRUE", "True", "true", "1")
d$n_group1 <- as.numeric(d$n_group1)
d$n_group2 <- as.numeric(d$n_group2)
d$effect_for_meta <- as.numeric(d$effect_for_meta)
d$sample_label <- paste(d$dataset, d$sample, sep = "__")
d <- d[d$included_in_meta & is.finite(d$effect_for_meta) &
         is.finite(d$n_group1) & is.finite(d$n_group2) &
         (d$n_group1 + d$n_group2) > 3, ]
d$effect_for_meta <- pmax(pmin(d$effect_for_meta, 0.999999), -0.999999)
d$yi <- atanh(d$effect_for_meta)
d$vi <- 1 / (d$n_group1 + d$n_group2 - 3)

rows <- list()
for (variant in unique(d$variant)) {
  for (contrast in unique(d$contrast)) {
    for (mp in unique(d$MP_id)) {
      g <- d[d$variant == variant & d$contrast == contrast & d$MP_id == mp, ]
      if (nrow(g) < 2) next
      fit <- rma(yi = g$yi, vi = g$vi, method = "REML")
      pooled_z <- as.numeric(fit$b[1])
      pooled <- tanh(pooled_z)
      ci <- tanh(c(fit$ci.lb, fit$ci.ub))
      pooled_sign <- sign(pooled)
      signs <- sign(g$effect_for_meta)
      direction <- if (pooled_sign == 0) mean(signs == 0) else mean(signs == pooled_sign)
      rows[[length(rows) + 1]] <- data.frame(
        variant = variant,
        contrast = contrast,
        MP_id = mp,
        MP_label = g$MP_label[1],
        pooled_rank_biserial = pooled,
        ci_low = ci[1],
        ci_high = ci[2],
        p_value = fit$pval,
        tau2 = fit$tau2,
        I2 = fit$I2,
        n_samples_included = nrow(g),
        direction_consistency = direction,
        variance_note = "Fisher-z variance uses raw contrast group count: 1/(n_group1+n_group2-3); anti-conservative under spatial autocorrelation. Variant S additionally carries twice-smoothing inflation.",
        stringsAsFactors = FALSE
      )

      png(file.path(plots_dir, paste0("forest_", variant, "_", contrast, "_", mp, ".png")),
          width = 1200, height = max(760, 38 * nrow(g) + 260), res = 150)
      par(mar = c(4.5, 8.5, 3.5, 2))
      forest(
        fit,
        slab = g$sample_label,
        transf = tanh,
        refline = 0,
        xlab = "Rank-biserial correlation",
        main = paste(variant, contrast, mp, g$MP_label[1], sep = " | "),
        cex = 0.78
      )
      dev.off()
    }
  }
}

summary <- do.call(rbind, rows)
if (is.null(summary) || !nrow(summary)) stop("No LISA contrast meta-analysis rows were produced")
summary <- summary[order(summary$variant, summary$contrast, summary$MP_id), ]
write.csv(summary, file.path(tables_dir, "snai1ac_lisa_meta_summary.csv"), row.names = FALSE)

purity_path <- file.path(tables_dir, "snai1ac_lisa_purity_diagnostic.csv")
if (file.exists(purity_path)) {
  p <- read.csv(purity_path, stringsAsFactors = FALSE)
  p <- p[is.finite(p$spearman_r) & is.finite(p$n_tumour_spots) & p$n_tumour_spots > 3, ]
  p$spearman_r <- pmax(pmin(p$spearman_r, 0.999999), -0.999999)
  p$yi <- atanh(p$spearman_r)
  p$vi <- 1 / (p$n_tumour_spots - 3)
  prow <- list()
  for (variant in unique(p$variant)) {
    g <- p[p$variant == variant, ]
    if (nrow(g) < 2) next
    fit <- rma(yi = g$yi, vi = g$vi, method = "REML")
    ci <- tanh(c(fit$ci.lb, fit$ci.ub))
    pooled <- tanh(as.numeric(fit$b[1]))
    signs <- sign(g$spearman_r)
    pooled_sign <- sign(pooled)
    direction <- if (pooled_sign == 0) mean(signs == 0) else mean(signs == pooled_sign)
    prow[[length(prow) + 1]] <- data.frame(
      variant = variant,
      pooled_spearman_r = pooled,
      ci_low = ci[1],
      ci_high = ci[2],
      p_value = fit$pval,
      tau2 = fit$tau2,
      I2 = fit$I2,
      n_samples_included = nrow(g),
      direction_consistency = direction,
      stringsAsFactors = FALSE
    )
  }
  if (length(prow)) write.csv(do.call(rbind, prow), file.path(tables_dir, "snai1ac_lisa_purity_meta_summary.csv"), row.names = FALSE)
}

cat("R version:", R.version.string, "\n")
cat("metafor:", as.character(packageVersion("metafor")), "\n")
cat("meta_rows:", nrow(summary), "\n")
