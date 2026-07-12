args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("Usage: Rscript snai1ac_mp_meta_analysis.R <output_dir>")

suppressPackageStartupMessages(library(metafor))

out_dir <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
tables_dir <- file.path(out_dir, "tables")
plots_dir <- file.path(out_dir, "plots")
dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

input <- file.path(tables_dir, "snai1ac_mp_per_sample_correlations.csv")
if (!file.exists(input)) stop("Missing per-sample correlation table: ", input)
d <- read.csv(input, stringsAsFactors = FALSE)
need <- c("arm", "MP_id", "MP_label", "sample", "dataset", "n_spots_used", "spearman_r", "included_in_meta")
missing <- setdiff(need, names(d))
if (length(missing)) stop("Per-sample table lacks columns: ", paste(missing, collapse = ", "))

d$included_in_meta <- as.character(d$included_in_meta) %in% c("TRUE", "True", "true", "1")
d$n_spots_used <- as.numeric(d$n_spots_used)
d$spearman_r <- as.numeric(d$spearman_r)
d$sample_label <- paste(d$dataset, d$sample, sep = "__")
d <- d[is.finite(d$spearman_r) & is.finite(d$n_spots_used) & d$n_spots_used > 3 & d$included_in_meta, ]
d$spearman_r <- pmax(pmin(d$spearman_r, 0.999999), -0.999999)
d$yi <- atanh(d$spearman_r)
d$vi <- 1 / (d$n_spots_used - 3)

arms <- unique(d$arm)
mps <- unique(d$MP_id)
rows <- list()

for (arm in arms) {
  for (mp in mps) {
    g <- d[d$arm == arm & d$MP_id == mp, ]
    if (nrow(g) < 2) next
    fit <- rma(yi = g$yi, vi = g$vi, method = "REML")
    pooled_z <- as.numeric(fit$b[1])
    pooled_r <- tanh(pooled_z)
    ci <- tanh(c(fit$ci.lb, fit$ci.ub))
    signs <- sign(g$spearman_r)
    pooled_sign <- sign(pooled_r)
    direction <- if (pooled_sign == 0) mean(signs == 0) else mean(signs == pooled_sign)
    rows[[length(rows) + 1]] <- data.frame(
      arm = arm,
      MP_id = mp,
      MP_label = g$MP_label[1],
      pooled_r = pooled_r,
      ci_low = ci[1],
      ci_high = ci[2],
      p_value = fit$pval,
      tau2 = fit$tau2,
      I2 = fit$I2,
      n_samples_included = nrow(g),
      direction_consistency = direction,
      variance_note = "Fisher-z variance uses raw per-arm spot count: 1/(n_spots_used-3); anti-conservative under spatial autocorrelation.",
      stringsAsFactors = FALSE
    )

    png(file.path(plots_dir, paste0("forest_", arm, "_", mp, ".png")), width = 1200, height = max(760, 38 * nrow(g) + 260), res = 150)
    par(mar = c(4.5, 8.5, 3.5, 2))
    forest(
      fit,
      slab = g$sample_label,
      transf = tanh,
      refline = 0,
      xlab = "Spearman r",
      main = paste(arm, mp, g$MP_label[1], sep = " | "),
      cex = 0.78
    )
    dev.off()
  }
}

summary <- do.call(rbind, rows)
if (is.null(summary) || !nrow(summary)) stop("No meta-analysis rows were produced")
summary <- summary[order(summary$MP_id, summary$arm), ]
write.csv(summary, file.path(tables_dir, "snai1ac_mp_meta_analysis_summary.csv"), row.names = FALSE)

arm_order <- c("arm1_raw_all_spots", "arm2_partial_all_spots_malignant", "arm3_tumour_only_rescored")
mp_order <- unique(summary$MP_id[order(summary$MP_id)])
mat <- matrix(NA_real_, nrow = length(mp_order), ncol = length(arm_order), dimnames = list(mp_order, arm_order))
labels <- setNames(summary$MP_label, summary$MP_id)
for (i in seq_len(nrow(summary))) mat[summary$MP_id[i], summary$arm[i]] <- summary$pooled_r[i]

png(file.path(plots_dir, "overview_heatmap_pooled_r.png"), width = 980, height = 760, res = 150)
par(mar = c(8, 12, 3, 5))
pal <- colorRampPalette(c("#2166AC", "white", "#B2182B"))(101)
breaks <- seq(-max(abs(mat), na.rm = TRUE), max(abs(mat), na.rm = TRUE), length.out = 102)
image(
  x = seq_len(ncol(mat)),
  y = seq_len(nrow(mat)),
  z = t(mat[nrow(mat):1, ]),
  col = pal,
  breaks = breaks,
  axes = FALSE,
  xlab = "",
  ylab = "",
  main = "Pooled SNAI1-ac x MP Spearman r"
)
axis(1, at = seq_len(ncol(mat)), labels = colnames(mat), las = 2, cex.axis = 0.78)
axis(2, at = seq_len(nrow(mat)), labels = paste(rev(mp_order), labels[rev(mp_order)], sep = " "), las = 2, cex.axis = 0.78)
for (i in seq_len(nrow(mat))) {
  for (j in seq_len(ncol(mat))) {
    val <- mat[rev(seq_len(nrow(mat)))[i], j]
    if (is.finite(val)) text(j, i, sprintf("%.2f", val), cex = 0.85)
  }
}
box()
dev.off()

cat("R version:", R.version.string, "\n")
cat("metafor:", as.character(packageVersion("metafor")), "\n")
cat("meta_rows:", nrow(summary), "\n")
