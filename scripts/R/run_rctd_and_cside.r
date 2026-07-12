# ==============================================================================
# Batch RCTD Deconvolution + C-SIDE for all Visium samples
# ==============================================================================
# Purpose: Run RCTD (cell type deconvolution) and C-SIDE (cell-type-specific DE)
#          across all exported Visium samples. Processes one sample at a time
#          with memory cleanup between samples.
#
# Models:
#   Model 1: SNAI1-ac only (run.CSIDE.single)
#   Model 2: SNAI1-ac + Malignant fraction (run.CSIDE with design matrix)
#
# Error handling:
#   - Three nested tryCatch blocks (RCTD, Model 1, Model 2)
#   - If RCTD fails → skip both models, move to next sample
#   - If Model 1 fails → still attempt Model 2
#   - If Model 2 fails → log error, move on
#   - Granular skip logic: each step checks for its own output files
#
# Input:
#   - GSE184880_annotated.rds (scRNA-seq reference, loaded once)
#   - Per sample: counts.csv.gz, coords.csv, metadata.csv
#
# Output per sample:
#   - rctd_weights.csv
#   - rctd_object.rds
#   - cside_all_results.csv, cside_significant.csv
#   - cside_2cov_all_results.csv, cside_2cov_significant.csv
#   - rctd_cside_object.rds, rctd_cside_2cov_object.rds
# ==============================================================================

library(spacexr)
library(Seurat)
library(Matrix)

set.seed(42)

# --- Configuration ---
input_base  <- "D:/HGSOC_Spatial_Atlas/scRNA_reference/rctd_inputs"
output_base <- "D:/HGSOC_Spatial_Atlas/scRNA_reference/rctd_outputs"
ref_path    <- "D:/HGSOC_Spatial_Atlas/scRNA_reference/GSE184880_annotated.rds"

datasets <- c("denisenko_2022", "ju_2024", "yamamoto_2025")

# C-SIDE parameters (same across all samples for comparability)
CELL_TYPE_THRESHOLD <- 50
WEIGHT_THRESHOLD    <- 0.8
MAX_CORES           <- 10

# ==============================================================================
# PART 0: Build scRNA-seq reference (once)
# ==============================================================================
cat("=== Loading scRNA-seq reference ===\n")

ref_seurat <- readRDS(ref_path)
cat("Loaded reference:", ncol(ref_seurat), "cells\n")
cat("Cell types:", paste(sort(unique(ref_seurat$cell_type)), collapse = ", "), "\n")

counts_ref <- GetAssayData(ref_seurat, layer = "counts")
cell_types <- as.factor(ref_seurat$cell_type)
names(cell_types) <- colnames(ref_seurat)
nUMI_ref <- colSums(counts_ref)
reference <- Reference(counts_ref, cell_types, nUMI_ref)

# Free Seurat object, keep only RCTD Reference
rm(ref_seurat, counts_ref, cell_types, nUMI_ref)
gc()

cat("Reference object ready.\n\n")

# ==============================================================================
# Helper: Extract and save C-SIDE results
# ==============================================================================
extract_cside_results <- function(rctd_obj, output_dir, prefix = "cside") {
  
  cside_cts <- names(rctd_obj@de_results$all_gene_list)
  cat("  Cell types analyzed:", paste(cside_cts, collapse = ", "), "\n")
  
  # All tested genes
  all_de <- list()
  for (ct in cside_cts) {
    de <- rctd_obj@de_results$all_gene_list[[ct]]
    if (!is.null(de) && nrow(de) > 0) {
      de$cell_type <- ct
      de$gene <- rownames(de)
      all_de[[ct]] <- de
      cat(sprintf("    %s: %d genes tested\n", ct, nrow(de)))
    }
  }
  combined_de <- do.call(rbind, all_de)
  rownames(combined_de) <- NULL
  write.csv(combined_de, file.path(output_dir, paste0(prefix, "_all_results.csv")),
            row.names = FALSE)
  
  # Significant genes (from sig_gene_list)
  cat("  Significant genes:\n")
  all_sig <- list()
  for (ct in cside_cts) {
    sig <- rctd_obj@de_results$sig_gene_list[[ct]]
    if (!is.null(sig) && nrow(sig) > 0) {
      sig$cell_type <- ct
      sig$gene <- rownames(sig)
      all_sig[[ct]] <- sig
      cat(sprintf("    %s: %d significant\n", ct, nrow(sig)))
    } else {
      cat(sprintf("    %s: 0 significant\n", ct))
    }
  }
  
  if (length(all_sig) > 0) {
    combined_sig <- do.call(rbind, all_sig)
    rownames(combined_sig) <- NULL
    write.csv(combined_sig, file.path(output_dir, paste0(prefix, "_significant.csv")),
              row.names = FALSE)
    cat(sprintf("  Saved: %s_significant.csv (%d genes)\n", prefix, nrow(combined_sig)))
  } else {
    cat(sprintf("  No significant genes for %s.\n", prefix))
  }
  
  cat(sprintf("  Saved: %s_all_results.csv\n", prefix))
}

# ==============================================================================
# PART 1: Auto-discover samples and process
# ==============================================================================
cat("=== Discovering samples ===\n")

# Build sample list
sample_list <- data.frame(dataset = character(), sample = character(),
                          stringsAsFactors = FALSE)
for (ds in datasets) {
  ds_dir <- file.path(input_base, ds)
  if (dir.exists(ds_dir)) {
    samples <- list.dirs(ds_dir, recursive = FALSE, full.names = FALSE)
    if (length(samples) > 0) {
      sample_list <- rbind(sample_list,
                           data.frame(dataset = ds, sample = samples,
                                      stringsAsFactors = FALSE))
    }
  }
}

cat(sprintf("Found %d samples across %d datasets:\n", nrow(sample_list),
            length(unique(sample_list$dataset))))
for (i in seq_len(nrow(sample_list))) {
  cat(sprintf("  [%d] %s / %s\n", i, sample_list$dataset[i], sample_list$sample[i]))
}
cat("\n")

# ==============================================================================
# PART 2: Process each sample
# ==============================================================================

for (i in seq_len(nrow(sample_list))) {
  
  ds   <- sample_list$dataset[i]
  samp <- sample_list$sample[i]
  
  cat(sprintf("\n###############################################################\n"))
  cat(sprintf("### Sample %d/%d: %s / %s\n", i, nrow(sample_list), ds, samp))
  cat(sprintf("###############################################################\n\n"))
  
  visium_dir <- file.path(input_base, ds, samp)
  output_dir <- file.path(output_base, ds, samp)
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # --- Verify input files exist ---
  required_files <- c("counts.csv.gz", "coords.csv", "metadata.csv")
  missing <- required_files[!file.exists(file.path(visium_dir, required_files))]
  if (length(missing) > 0) {
    cat("  MISSING input files:", paste(missing, collapse = ", "), "— skipping.\n")
    next
  }
  
  # Track success
  rctd_ok <- FALSE
  
  # -----------------------------------------------------------------------
  # RCTD (skip if already done, reload from disk)
  # -----------------------------------------------------------------------
  if (file.exists(file.path(output_dir, "rctd_object.rds"))) {
    cat("  RCTD already done — loading from disk...\n")
    rctd <- readRDS(file.path(output_dir, "rctd_object.rds"))
    weights <- as.data.frame(rctd@results$weights)
    cat(sprintf("    %d spots x %d cell types\n", nrow(weights), ncol(weights)))
    rctd_ok <- TRUE
  } else {
    tryCatch({
      # Load Visium data
      cat("  Loading Visium data...\n")
      counts_vis <- read.csv(file.path(visium_dir, "counts.csv.gz"),
                             row.names = 1, check.names = FALSE)
      counts_vis <- as(t(as.matrix(counts_vis)), "dgCMatrix")
      cat(sprintf("    Counts: %d genes x %d spots\n", nrow(counts_vis), ncol(counts_vis)))
      
      coords <- read.csv(file.path(visium_dir, "coords.csv"), row.names = 1)
      colnames(coords) <- c("x", "y")
      
      nUMI_vis <- colSums(counts_vis)
      spatial_rna <- SpatialRNA(coords, counts_vis, nUMI_vis)
      rm(counts_vis, coords, nUMI_vis)
      gc()
      
      # Run RCTD
      cat("  Running RCTD (full mode)...\n")
      rctd <- create.RCTD(spatial_rna, reference, max_cores = MAX_CORES)
      rctd <- run.RCTD(rctd, doublet_mode = "full")
      rm(spatial_rna)
      gc()
      
      weights <- as.data.frame(rctd@results$weights)
      cat(sprintf("    %d spots x %d cell types\n", nrow(weights), ncol(weights)))
      write.csv(weights, file.path(output_dir, "rctd_weights.csv"))
      saveRDS(rctd, file.path(output_dir, "rctd_object.rds"))
      rctd_ok <- TRUE
      
    }, error = function(e) {
      cat(sprintf("  RCTD ERROR: %s\n", e$message))
    })
  }
  
  if (!rctd_ok) {
    cat("  Skipping C-SIDE (RCTD failed).\n")
    rm(list = intersect(c("rctd", "weights", "spatial_rna"), ls()))
    gc()
    next
  }
  
  # -----------------------------------------------------------------------
  # Prepare covariates
  # -----------------------------------------------------------------------
  metadata <- read.csv(file.path(visium_dir, "metadata.csv"), row.names = 1)
  cat(sprintf("    SNAI1-ac range: [%.3f, %.3f]\n",
              min(metadata$SNAI1.ac_score, na.rm = TRUE),
              max(metadata$SNAI1.ac_score, na.rm = TRUE)))
  
  snai1_score <- metadata$SNAI1.ac_score
  names(snai1_score) <- rownames(metadata)
  common_spots <- intersect(names(snai1_score), rownames(weights))
  snai1_score <- snai1_score[common_spots]
  snai1_score <- snai1_score[!is.na(snai1_score)]
  cat(sprintf("    %d spots with RCTD + SNAI1-ac\n", length(snai1_score)))
  
  # -----------------------------------------------------------------------
  # C-SIDE Model 1: SNAI1-ac only (skip if already done)
  # -----------------------------------------------------------------------
  if (file.exists(file.path(output_dir, "cside_all_results.csv"))) {
    cat("  Model 1 already done — skipping.\n")
  } else {
    tryCatch({
      cat("  Running C-SIDE Model 1 (SNAI1-ac only)...\n")
      rctd <- run.CSIDE.single(rctd,
                               snai1_score,
                               doublet_mode = FALSE,
                               cell_type_threshold = CELL_TYPE_THRESHOLD,
                               weight_threshold = WEIGHT_THRESHOLD)
      cat("  Model 1 results:\n")
      extract_cside_results(rctd, output_dir, prefix = "cside")
      saveRDS(rctd, file.path(output_dir, "rctd_cside_object.rds"))
    }, error = function(e) {
      cat(sprintf("  Model 1 ERROR: %s\n", e$message))
    })
  }
  
  # -----------------------------------------------------------------------
  # C-SIDE Model 2: SNAI1-ac + Malignant fraction (skip if already done)
  # -----------------------------------------------------------------------
  if (file.exists(file.path(output_dir, "cside_2cov_all_results.csv"))) {
    cat("  Model 2 already done — skipping.\n")
  } else {
    tryCatch({
      cat("  Running C-SIDE Model 2 (SNAI1-ac + Malignant)...\n")
      barcodes <- names(snai1_score)
      
      snai1_norm <- (snai1_score - min(snai1_score)) / (max(snai1_score) - min(snai1_score))
      
      mal_frac <- metadata$Malignant
      names(mal_frac) <- rownames(metadata)
      mal_frac <- mal_frac[barcodes]
      mal_norm <- (mal_frac - min(mal_frac)) / (max(mal_frac) - min(mal_frac))
      
      X <- cbind(1, snai1_norm, mal_norm)
      rownames(X) <- barcodes
      
      cat(sprintf("    Correlation SNAI1-ac vs Malignant: %.3f\n",
                  cor(snai1_norm, mal_norm)))
      
      # Reload clean RCTD object (Model 1 may have modified it)
      rctd_clean <- readRDS(file.path(output_dir, "rctd_object.rds"))
      
      rctd_2cov <- run.CSIDE(rctd_clean,
                             X,
                             barcodes,
                             cell_type_threshold = CELL_TYPE_THRESHOLD,
                             doublet_mode = FALSE,
                             weight_threshold = WEIGHT_THRESHOLD,
                             params_to_test = 2,
                             test_mode = "individual")
      
      cat("  Model 2 results:\n")
      extract_cside_results(rctd_2cov, output_dir, prefix = "cside_2cov")
      saveRDS(rctd_2cov, file.path(output_dir, "rctd_cside_2cov_object.rds"))
      
      rm(rctd_clean, rctd_2cov)
      
    }, error = function(e) {
      cat(sprintf("  Model 2 ERROR: %s\n", e$message))
    })
  }
  
  # -----------------------------------------------------------------------
  # Cleanup (only remove what exists)
  # -----------------------------------------------------------------------
  cat("  Cleaning up memory...\n")
  cleanup_vars <- c("rctd", "rctd_clean", "rctd_2cov", "weights", "metadata",
                    "snai1_score", "snai1_norm", "mal_frac", "mal_norm",
                    "X", "barcodes", "common_spots")
  rm(list = intersect(cleanup_vars, ls()))
  gc()
  
  cat(sprintf("  Done with %s / %s\n", ds, samp))
}

cat("\n=== ALL SAMPLES COMPLETE ===\n")

# Quick summary across all samples
output_base <- "D:/HGSOC_Spatial_Atlas/scRNA_reference/rctd_outputs"

for (ds in c("denisenko_2022", "ju_2024", "yamamoto_2025")) {
  ds_dir <- file.path(output_base, ds)
  if (!dir.exists(ds_dir)) next
  samples <- list.dirs(ds_dir, recursive = FALSE, full.names = FALSE)
  for (samp in samples) {
    out <- file.path(ds_dir, samp)
    m1 <- file.path(out, "cside_significant.csv")
    m2 <- file.path(out, "cside_2cov_significant.csv")
    
    n_m1 <- if (file.exists(m1)) nrow(read.csv(m1)) else NA
    n_m2 <- if (file.exists(m2)) nrow(read.csv(m2)) else NA
    
    cat(sprintf("%s / %s : Model1=%s  Model2=%s\n", 
                ds, samp, 
                ifelse(is.na(n_m1), "FAILED", as.character(n_m1)),
                ifelse(is.na(n_m2), "FAILED", as.character(n_m2))))
  }
}

# combine DEG outputsinto one csv file - Model 2

output_base <- "D:/HGSOC_Spatial_Atlas/scRNA_reference/rctd_outputs"
all_sig <- list()

for (ds in c("denisenko_2022", "ju_2024", "yamamoto_2025")) {
  ds_dir <- file.path(output_base, ds)
  if (!dir.exists(ds_dir)) next
  samples <- list.dirs(ds_dir, recursive = FALSE, full.names = FALSE)
  for (samp in samples) {
    f <- file.path(ds_dir, samp, "cside_2cov_significant.csv")
    if (file.exists(f)) {
      df <- read.csv(f)
      df$dataset <- ds
      df$sample <- samp
      all_sig[[paste(ds, samp)]] <- df
    }
  }
}

combined <- do.call(rbind, all_sig)
rownames(combined) <- NULL
write.csv(combined, file.path(output_base, "all_samples_2cov_significant.csv"), row.names = FALSE)
cat("Combined:", nrow(combined), "rows across", length(all_sig), "samples\n")
cat("Unique genes:", length(unique(combined$gene)), "\n")
cat("Cell types:", paste(sort(unique(combined$cell_type)), collapse = ", "), "\n")

# combine DEG outputsinto one csv file - Model 1

all_sig_m1 <- list()
for (ds in c("denisenko_2022", "ju_2024", "yamamoto_2025")) {
  ds_dir <- file.path(output_base, ds)
  if (!dir.exists(ds_dir)) next
  samples <- list.dirs(ds_dir, recursive = FALSE, full.names = FALSE)
  for (samp in samples) {
    f <- file.path(ds_dir, samp, "cside_significant.csv")
    if (file.exists(f)) {
      df <- read.csv(f)
      df$dataset <- ds
      df$sample <- samp
      all_sig_m1[[paste(ds, samp)]] <- df
    }
  }
}

combined_m1 <- do.call(rbind, all_sig_m1)
rownames(combined_m1) <- NULL
write.csv(combined_m1, file.path(output_base, "all_samples_m1_significant.csv"), row.names = FALSE)
cat("Combined M1:", nrow(combined_m1), "rows across", length(all_sig_m1), "samples\n")

# combine all tested genes per cell type from Model 2 into one csv 

output_base <- "D:/HGSOC_Spatial_Atlas/scRNA_reference/rctd_outputs"
all_results <- list()

for (ds in c("denisenko_2022", "ju_2024", "yamamoto_2025")) {
  ds_dir <- file.path(output_base, ds)
  if (!dir.exists(ds_dir)) next
  samples <- list.dirs(ds_dir, recursive = FALSE, full.names = FALSE)
  for (samp in samples) {
    f <- file.path(ds_dir, samp, "cside_2cov_all_results.csv")
    if (file.exists(f)) {
      df <- read.csv(f)
      df$dataset <- ds
      df$sample <- samp
      all_results[[paste(ds, samp)]] <- df
      cat(sprintf("  %s / %s: %d rows\n", ds, samp, nrow(df)))
    }
  }
}

combined_all <- do.call(rbind, all_results)
rownames(combined_all) <- NULL
write.csv(combined_all, file.path(output_base, "all_samples_2cov_all_results.csv"), row.names = FALSE)
cat(sprintf("\nCombined: %d rows across %d samples\n", nrow(combined_all), length(all_results)))

# For each cell type, how many samples was each gene tested in?

df <- read.csv("D:/HGSOC_Spatial_Atlas/scRNA_reference/rctd_outputs/all_samples_2cov_all_results.csv")

for (ct in unique(df$cell_type)) {
  sub <- df[df$cell_type == ct, ]
  gene_counts <- table(sub$gene)
  cat(sprintf("\n%s: %d genes total\n", ct, length(gene_counts)))
  print(quantile(gene_counts, c(0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)))
}
