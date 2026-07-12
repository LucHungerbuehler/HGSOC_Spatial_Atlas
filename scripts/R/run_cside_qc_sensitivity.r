# Run C-SIDE QC sensitivity from existing clean RCTD objects.
# Model: SNAI1-ac + malignant fraction + available QC covariates.

library(spacexr)

input_meta_base <- "D:/HGSOC_Spatial_Atlas/scRNA_reference/rctd_inputs_qc_sensitivity"
original_rctd_base <- "D:/HGSOC_Spatial_Atlas/scRNA_reference/rctd_outputs"
output_base <- "D:/HGSOC_Spatial_Atlas/scRNA_reference/rctd_outputs_qc_sensitivity"

datasets <- c("denisenko_2022", "ju_2024", "yamamoto_2025")
CELL_TYPE_THRESHOLD <- 50
WEIGHT_THRESHOLD <- 0.8

args <- commandArgs(trailingOnly = TRUE)
only_dataset <- NA_character_
only_sample <- NA_character_
extract_existing <- FALSE
if (length(args) > 0) {
  for (i in seq_along(args)) {
    if (args[[i]] == "--dataset" && i < length(args)) {
      only_dataset <- args[[i + 1]]
    }
    if (args[[i]] == "--sample" && i < length(args)) {
      only_sample <- args[[i + 1]]
    }
    if (args[[i]] == "--extract-existing") {
      extract_existing <- TRUE
    }
  }
}
if (!is.na(only_dataset)) {
  datasets <- intersect(datasets, only_dataset)
}

log_msg <- function(...) {
  cat(sprintf("[%s] %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), sprintf(...)))
  flush.console()
}

write_status <- function(out_dir, dataset, sample, status, detail = "") {
  status_path <- file.path(out_dir, "cside_qc_status.tsv")
  line <- paste(
    format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    dataset,
    sample,
    status,
    detail,
    sep = "\t"
  )
  write(line, file = status_path, append = TRUE)
}

extract_cside_results <- function(rctd_obj, output_dir, prefix = "cside_qc") {
  skipped_path <- file.path(output_dir, paste0(prefix, "_extract_skipped_celltypes.csv"))
  sig_path <- file.path(output_dir, paste0(prefix, "_significant.csv"))
  if (file.exists(skipped_path)) unlink(skipped_path)
  if (file.exists(sig_path)) unlink(sig_path)

  if (is.null(rctd_obj@de_results$all_gene_list)) {
    stop("C-SIDE object does not contain de_results$all_gene_list")
  }
  cside_cts <- names(rctd_obj@de_results$all_gene_list)
  log_msg("Extracting C-SIDE results for cell types: %s", paste(cside_cts, collapse = ", "))

  all_res <- list()
  skipped <- list()
  for (ct in cside_cts) {
    df <- rctd_obj@de_results$all_gene_list[[ct]]
    if (is.null(df)) {
      skipped[[ct]] <- data.frame(cell_type = ct, reason = "null")
      next
    }
    if (!is.data.frame(df) && !is.matrix(df)) {
      skipped[[ct]] <- data.frame(cell_type = ct, reason = paste0("class_", paste(class(df), collapse = "_")))
      next
    }
    df <- as.data.frame(df)
    if (is.na(nrow(df)) || nrow(df) == 0) {
      skipped[[ct]] <- data.frame(cell_type = ct, reason = "empty")
      next
    }
    df$cell_type <- ct
    df$gene <- rownames(df)
    all_res[[ct]] <- df
  }
  if (length(skipped) > 0) {
    write.csv(
      do.call(rbind, skipped),
      skipped_path,
      row.names = FALSE
    )
  }
  if (length(all_res) == 0) {
    stop("No non-empty C-SIDE de_results tables were available to extract")
  }
  combined <- do.call(rbind, all_res)
  write.csv(combined, file.path(output_dir, paste0(prefix, "_all_results.csv")), row.names = FALSE)

  all_sig <- list()
  if (!is.null(rctd_obj@de_results$sig_gene_list)) {
    for (ct in cside_cts) {
      sig <- rctd_obj@de_results$sig_gene_list[[ct]]
      if (!is.null(sig) && is.data.frame(sig) && nrow(sig) > 0) {
        sig$cell_type <- ct
        sig$gene <- rownames(sig)
        all_sig[[ct]] <- sig
      }
    }
  }
  if (length(all_sig) > 0) {
    sig_combined <- do.call(rbind, all_sig)
    write.csv(sig_combined, sig_path, row.names = FALSE)
  }
}

minmax <- function(x) {
  rng <- range(x, na.rm = TRUE)
  if (!is.finite(rng[1]) || !is.finite(rng[2]) || rng[1] == rng[2]) return(rep(0, length(x)))
  (x - rng[1]) / (rng[2] - rng[1])
}

for (ds in datasets) {
  ds_dir <- file.path(input_meta_base, ds)
  if (!dir.exists(ds_dir)) {
    log_msg("Skipping dataset %s: input metadata directory does not exist", ds)
    next
  }
  samples <- list.files(ds_dir)
  if (!is.na(only_sample)) {
    samples <- intersect(samples, only_sample)
  }
  log_msg("Dataset %s: %d sample(s) queued", ds, length(samples))
  for (samp in samples) {
    meta_path <- file.path(ds_dir, samp, "metadata.csv")
    rctd_path <- file.path(original_rctd_base, ds, samp, "rctd_object.rds")
    out_dir <- file.path(output_base, ds, samp)
    dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
    write_status(out_dir, ds, samp, "queued")
    if (!file.exists(meta_path) || !file.exists(rctd_path)) {
      log_msg("Skipping %s/%s: missing metadata or RCTD object", ds, samp)
      write_status(out_dir, ds, samp, "skipped", "missing_metadata_or_rctd")
      next
    }
    out_file <- file.path(out_dir, "cside_qc_all_results.csv")
    rds_file <- file.path(out_dir, "rctd_cside_qc_object.rds")
    if (extract_existing && file.exists(rds_file)) {
      log_msg("Extract-existing mode for %s/%s from %s", ds, samp, rds_file)
      write_status(out_dir, ds, samp, "extract_existing_rds_forced", rds_file)
      fit <- readRDS(rds_file)
      tryCatch(
        extract_cside_results(fit, out_dir, prefix = "cside_qc"),
        error = function(e) {
          log_msg("ERROR extracting existing C-SIDE results for %s/%s: %s", ds, samp, e$message)
          write_status(out_dir, ds, samp, "error_extract_existing_rds", e$message)
        }
      )
      if (file.exists(out_file)) {
        write_status(out_dir, ds, samp, "done_extract_existing_rds", out_file)
      }
      next
    }
    if (file.exists(out_file)) {
      log_msg("Skipping %s/%s: existing output %s", ds, samp, out_file)
      write_status(out_dir, ds, samp, "skipped", "output_exists")
      next
    }

    log_msg("Reading metadata for %s/%s", ds, samp)
    metadata <- read.csv(meta_path, row.names = 1, check.names = FALSE)
    log_msg("Reading clean RCTD object for %s/%s", ds, samp)
    rctd <- readRDS(rctd_path)
    barcodes <- rownames(metadata)
    if (file.exists(rds_file) && !file.exists(out_file)) {
      log_msg("Found existing QC C-SIDE object for %s/%s; extracting results without rerun", ds, samp)
      write_status(out_dir, ds, samp, "extracting_existing_rds", rds_file)
      fit <- readRDS(rds_file)
      tryCatch(
        extract_cside_results(fit, out_dir, prefix = "cside_qc"),
        error = function(e) {
          log_msg("ERROR extracting existing C-SIDE results for %s/%s: %s", ds, samp, e$message)
          write_status(out_dir, ds, samp, "error_extract_existing_rds", e$message)
        }
      )
      if (file.exists(out_file)) {
        write_status(out_dir, ds, samp, "done_extract_existing_rds", out_file)
      }
      next
    }

    base_required <- c("SNAI1-ac_score", "Malignant")
    if (!all(base_required %in% colnames(metadata))) {
      log_msg("Skipping %s/%s: missing base columns", ds, samp)
      write_status(out_dir, ds, samp, "skipped", "missing_base_columns")
      next
    }
    qc_candidates <- c(
      "total_counts_rctd_input",
      "n_genes_by_counts_rctd_input",
      "pct_mito_rctd_input"
    )
    qc_available <- c()
    for (covar in qc_candidates) {
      if (!covar %in% colnames(metadata)) next
      x <- metadata[[covar]]
      if (sum(is.finite(x)) < 3) next
      if (sd(x, na.rm = TRUE) == 0) next
      qc_available <- c(qc_available, covar)
    }
    required <- c(base_required, qc_available)
    keep <- complete.cases(metadata[, required, drop = FALSE])
    metadata <- metadata[keep, , drop = FALSE]
    barcodes <- rownames(metadata)
    if (nrow(metadata) < 20) {
      log_msg("Skipping %s/%s: only %d complete spots", ds, samp, nrow(metadata))
      write_status(out_dir, ds, samp, "skipped", "too_few_complete_spots")
      next
    }

    x_parts <- list(
      intercept = rep(1, nrow(metadata)),
      snai1_norm = minmax(metadata[["SNAI1-ac_score"]]),
      mal_norm = minmax(metadata[["Malignant"]])
    )
    if ("total_counts_rctd_input" %in% qc_available) {
      x_parts[["log_total_counts_norm"]] <- minmax(log1p(metadata[["total_counts_rctd_input"]]))
    }
    if ("n_genes_by_counts_rctd_input" %in% qc_available) {
      x_parts[["n_genes_norm"]] <- minmax(metadata[["n_genes_by_counts_rctd_input"]])
    }
    if ("pct_mito_rctd_input" %in% qc_available) {
      x_parts[["pct_mito_norm"]] <- minmax(metadata[["pct_mito_rctd_input"]])
    }
    X <- do.call(cbind, x_parts)
    rownames(X) <- barcodes

    detail <- sprintf(
      "%d spots; %d covariates; QC covariates: %s",
      nrow(X), ncol(X), paste(qc_available, collapse = ", ")
    )
    log_msg(
      "Running QC C-SIDE for %s/%s (%s)",
      ds, samp, detail
    )
    write_status(out_dir, ds, samp, "running_run_cside", detail)
    started <- Sys.time()
    fit <- tryCatch(
      run.CSIDE(
        rctd,
        X,
        barcodes,
        cell_type_threshold = CELL_TYPE_THRESHOLD,
        doublet_mode = FALSE,
        weight_threshold = WEIGHT_THRESHOLD,
        params_to_test = 2,
        test_mode = "individual"
      ),
      error = function(e) {
        log_msg("ERROR in run.CSIDE for %s/%s: %s", ds, samp, e$message)
        write_status(out_dir, ds, samp, "error_run_cside", e$message)
        return(NULL)
      }
    )
    if (is.null(fit)) next
    elapsed <- round(as.numeric(difftime(Sys.time(), started, units = "mins")), 2)
    log_msg("run.CSIDE complete for %s/%s after %.2f min", ds, samp, elapsed)
    saveRDS(fit, file.path(out_dir, "rctd_cside_qc_object.rds"))
    write_status(out_dir, ds, samp, "extracting_results", sprintf("elapsed_min=%.2f", elapsed))
    extract_ok <- FALSE
    tryCatch(
      {
        extract_cside_results(fit, out_dir, prefix = "cside_qc")
        extract_ok <<- file.exists(out_file)
      },
      error = function(e) {
        log_msg("ERROR extracting C-SIDE results for %s/%s: %s", ds, samp, e$message)
        write_status(out_dir, ds, samp, "error_extract_results", e$message)
      }
    )
    if (extract_ok) {
      log_msg("Saved QC C-SIDE outputs for %s/%s", ds, samp)
      write_status(out_dir, ds, samp, "done", sprintf("elapsed_min=%.2f", elapsed))
    }
  }
}
