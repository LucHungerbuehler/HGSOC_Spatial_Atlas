# ==============================================================================
# S1: ORA on Bulk RNA-seq DEG Lists (clusterProfiler)
# ==============================================================================
# Purpose: Run overrepresentation analysis on the 3 PEO4 bulk DEG comparisons
#          to understand what pathways the SNAI1 acetylation signature genes
#          (and the broader DEG lists) are enriched for.
#
# Comparisons (all PEO4):
#   1. SNAI1 vs GFP      — effect of acetylatable SNAI1 overexpression
#   2. 2R vs GFP          — effect of non-acetylatable SNAI1 overexpression
#   3. 2R vs SNAI1         — acetylation-specific effect (= signature source)
#
# Thresholds: BH-corrected padj < 0.05, |log2FC| > 1 (same as signature)
#
# Databases: MSigDB Hallmark, GO Biological Process, KEGG
#
# Input:
#   - tt_PEO4SNAI12R_Analysis.xlsx (Cmpr sheet)
#
# Output:
#   - ora_bulk_degs_all.csv        (combined results, all comparisons/directions/dbs)
#   - ora_deg_lists.csv            (the filtered DEG lists used as input)
#   - ora_summary.txt              (console summary)
# ==============================================================================

library(readxl)
library(clusterProfiler)
library(org.Hs.eg.db)
library(msigdbr)

set.seed(42)

# --- Configuration ---
input_file <- "C:/Users/luchu/Documents/MSc/Master Thesis/Code/RNA-seq/tt_PEO4-SNAI1-2R_Analysis.xlsx"
output_dir <- "D:/HGSOC_Spatial_Atlas/05_analysis_ready/bulk_ora"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Thresholds (same as signature derivation)
FDR_THRESHOLD  <- 0.05
LOG2FC_THRESHOLD <- 1.0

# --- Define comparisons ---
# Each comparison: name, p-value column, log2FC column
comparisons <- list(
  list(
    name    = "PEO4_SNAI1_vs_GFP",
    pval_col = "PEO4_SNAI1vsGFP_PPEE",
    fc_col   = "PEO4_lg2fc (SNAI1-GFP)"
  ),
  list(
    name    = "PEO4_2R_vs_GFP",
    pval_col = "PEO4-2R_SNAI1vsGFP_PPEE",
    fc_col   = "PEO4-2R_lg2fc (SNAI1-GFP)"
  ),
  list(
    name    = "PEO4_2R_vs_SNAI1",
    pval_col = "PEO4-2R_SNAI1vsSNAI1_PPEE",
    fc_col   = "PEO4-2R_lg2fc (SNAI1-SNAI1)"
  )
)

# ==============================================================================
# PART 1: Load data and prepare gene lists
# ==============================================================================
cat("=== Loading data ===\n")
df <- read_excel(input_file, sheet = "Cmpr")
cat(sprintf("Loaded: %d genes\n", nrow(df)))

# Get Hallmark gene sets for enricher()
cat("Loading MSigDB Hallmark gene sets...\n")
hallmark_df <- msigdbr(species = "Homo sapiens", category = "H")
hallmark_sets <- hallmark_df[, c("gs_name", "gene_symbol")]
# enricher() expects a data.frame with columns: term, gene
colnames(hallmark_sets) <- c("term", "gene")

# Background: all genes in the dataset (for proper ORA)
all_genes <- df$Gene
cat(sprintf("Background universe: %d genes\n\n", length(all_genes)))

# Fix: convert columns from character to numeric
for (comp in comparisons) {
  df[[comp$pval_col]] <- as.numeric(df[[comp$pval_col]])
  df[[comp$fc_col]]   <- as.numeric(df[[comp$fc_col]])
}
# ==============================================================================
# PART 2: Filter DEGs and run ORA per comparison x direction
# ==============================================================================
all_results <- list()
all_degs    <- list()
all_enrich_objects <- list()

for (comp in comparisons) {
  
  cat(sprintf("=== %s ===\n", comp$name))
  
  # Extract columns
  pvals_raw <- df[[comp$pval_col]]
  log2fc    <- df[[comp$fc_col]]
  genes     <- df$Gene
  
  # BH correction
  pvals_raw[is.na(pvals_raw)] <- 1
  padj <- p.adjust(pvals_raw, method = "BH")
  
  # Filter
  sig <- which(padj < FDR_THRESHOLD & abs(log2fc) > LOG2FC_THRESHOLD)
  cat(sprintf("  Significant DEGs (padj<%.2f, |log2FC|>%.1f): %d\n",
              FDR_THRESHOLD, LOG2FC_THRESHOLD, length(sig)))
  
  if (length(sig) == 0) {
    cat("  No DEGs — skipping.\n\n")
    next
  }
  
  # Split by direction
  sig_up   <- sig[log2fc[sig] > 0]
  sig_down <- sig[log2fc[sig] < 0]
  cat(sprintf("  UP: %d, DOWN: %d\n", length(sig_up), length(sig_down)))
  
  # Save DEG info
  for (idx in sig) {
    all_degs[[length(all_degs) + 1]] <- data.frame(
      comparison = comp$name,
      gene       = genes[idx],
      log2FC     = log2fc[idx],
      padj       = padj[idx],
      direction  = ifelse(log2fc[idx] > 0, "UP", "DOWN"),
      stringsAsFactors = FALSE
    )
  }
  
  # --- Run ORA for each direction ---
  for (dir_info in list(
    list(name = "UP",   indices = sig_up),
    list(name = "DOWN", indices = sig_down)
  )) {
    
    dir_name <- dir_info$name
    dir_idx  <- dir_info$indices
    
    if (length(dir_idx) < 3) {
      cat(sprintf("  %s: only %d genes — skipping ORA.\n", dir_name, length(dir_idx)))
      next
    }
    
    gene_list <- genes[dir_idx]
    cat(sprintf("\n  --- %s %s (%d genes) ---\n", comp$name, dir_name, length(gene_list)))
    
    # --- Hallmark (via enricher) ---
    tryCatch({
      ego_hall <- enricher(
        gene     = gene_list,
        universe = all_genes,
        TERM2GENE = hallmark_sets,
        pvalueCutoff = 1,       # keep all for exploration
        qvalueCutoff = 1,
        minGSSize = 10,
        maxGSSize = 500
      )
      if (!is.null(ego_hall) && nrow(ego_hall@result) > 0) {
        res <- ego_hall@result
        res$comparison <- comp$name
        res$direction  <- dir_name
        res$database   <- "Hallmark"
        all_results[[length(all_results) + 1]] <- res
        n_sig <- sum(res$p.adjust < 0.05)
        cat(sprintf("  Hallmark: %d tested, %d significant (padj<0.05)\n",
                    nrow(res), n_sig))
                    all_enrich_objects[[paste0(comp$name, "_", dir_name, "Hallmark")]] <- ego_hall
      }
    }, error = function(e) cat(sprintf("  Hallmark ERROR: %s\n", e$message)))
    
    # --- GO Biological Process ---
    tryCatch({
      ego_go <- enrichGO(
        gene     = gene_list,
        universe = all_genes,
        OrgDb    = org.Hs.eg.db,
        keyType  = "SYMBOL",
        ont      = "BP",
        pvalueCutoff = 1,
        qvalueCutoff = 1,
        minGSSize = 10,
        maxGSSize = 500,
        readable = FALSE
      )
      if (!is.null(ego_go) && nrow(ego_go@result) > 0) {
        res <- ego_go@result
        res$comparison <- comp$name
        res$direction  <- dir_name
        res$database   <- "GO_BP"
        all_results[[length(all_results) + 1]] <- res
        n_sig <- sum(res$p.adjust < 0.05)
        cat(sprintf("  GO_BP: %d tested, %d significant (padj<0.05)\n",
                    nrow(res), n_sig))
        all_enrich_objects[[paste0(comp$name, "_", dir_name, "GO_BP")]] <- ego_go
      }
    }, error = function(e) cat(sprintf("  GO_BP ERROR: %s\n", e$message)))
    
    # --- KEGG ---
    # KEGG requires Entrez IDs
    tryCatch({
      entrez_map <- bitr(gene_list, fromType = "SYMBOL", toType = "ENTREZID",
                         OrgDb = org.Hs.eg.db)
      entrez_bg  <- bitr(all_genes, fromType = "SYMBOL", toType = "ENTREZID",
                         OrgDb = org.Hs.eg.db)
      
      if (nrow(entrez_map) >= 3) {
        ego_kegg <- enrichKEGG(
          gene     = entrez_map$ENTREZID,
          universe = entrez_bg$ENTREZID,
          organism = "hsa",
          pvalueCutoff = 1,
          qvalueCutoff = 1,
          minGSSize = 10,
          maxGSSize = 500
        )
        if (!is.null(ego_kegg) && nrow(ego_kegg@result) > 0) {
          res <- ego_kegg@result
          res$comparison <- comp$name
          res$direction  <- dir_name
          res$database   <- "KEGG"
          all_results[[length(all_results) + 1]] <- res
          n_sig <- sum(res$p.adjust < 0.05)
          cat(sprintf("  KEGG: %d tested, %d significant (padj<0.05)\n",
                      nrow(res), n_sig))
          all_enrich_objects[[paste0(comp$name, "_", dir_name, "KEGG")]] <- ego_kegg
        }
      }
    }, error = function(e) cat(sprintf("  KEGG ERROR: %s\n", e$message)))
  }
  
  cat("\n")
}

# ==============================================================================
# PART 3: Combine and save
# ==============================================================================
cat("=== Saving results ===\n")

# Combined ORA results
if (length(all_results) > 0) {
  combined <- do.call(rbind, lapply(all_results, function(x){
    # Keep only columns common to all, plus our added ones
    keep <- c("ID", "Description", "GeneRatio", "BgRatio", 
              "pvalue", "p.adjust", "qvalue", "geneID", "Count",
              "comparison", "direction", "database")
    keep <- intersect(keep, colnames(x))
    x[,keep]
  }))
  write.csv(combined, file.path(output_dir, "ora_bulk_degs_all.csv"),
            row.names = FALSE)
  cat(sprintf("Saved: ora_bulk_degs_all.csv (%d rows)\n", nrow(combined)))
  
saveRDS(all_enrich_objects, file.path(output_dir, "ora_enrich_objects.rds"))
cat(sprintf("Saved: ora_enrich_objects.rds (%d objects)\n", length(all_enrich_objects)))

  # Summary
  cat("\n=== SUMMARY: Significant terms (padj < 0.05) ===\n")
  sig_combined <- combined[combined$p.adjust < 0.05, ]
  if (nrow(sig_combined) > 0) {
    for (comp_name in unique(sig_combined$comparison)) {
      for (dir_name in c("UP", "DOWN")) {
        for (db_name in c("Hallmark", "GO_BP", "KEGG")) {
          subset <- sig_combined[sig_combined$comparison == comp_name &
                                   sig_combined$direction == dir_name &
                                   sig_combined$database == db_name, ]
          if (nrow(subset) > 0) {
            cat(sprintf("\n%s | %s | %s (%d terms):\n", comp_name, dir_name, db_name, nrow(subset)))
            top_n <- min(nrow(subset), 5)
            for (j in seq_len(top_n)) {
              cat(sprintf("  %s (padj=%.2e, %s)\n",
                          subset$Description[j], subset$p.adjust[j], subset$GeneRatio[j]))
            }
          }
        }
      }
    }
  } else {
    cat("  No significant terms found.\n")
  }
} else {
  cat("  No ORA results generated.\n")
}

# DEG lists used
if (length(all_degs) > 0) {
  deg_df <- do.call(rbind, all_degs)
  write.csv(deg_df, file.path(output_dir, "ora_deg_lists.csv"), row.names = FALSE)
  cat(sprintf("\nSaved: ora_deg_lists.csv (%d DEGs total)\n", nrow(deg_df)))
  
  # Quick summary
  cat("\nDEG counts per comparison:\n")
  print(table(deg_df$comparison, deg_df$direction))
}

# Save console output
sink(file.path(output_dir, "ora_summary.txt"))
cat("ORA completed. See ora_bulk_degs_all.csv for full results.\n")
cat(sprintf("Thresholds: padj < %g, |log2FC| > %g\n", FDR_THRESHOLD, LOG2FC_THRESHOLD))
cat(sprintf("Databases: Hallmark, GO_BP, KEGG\n"))
if (length(all_degs) > 0) {
  cat("\nDEG counts:\n")
  print(table(deg_df$comparison, deg_df$direction))
}
sink()

cat("\n=== ALL DONE ===\n")
cat("Output directory:", output_dir, "\n")

# ==============================================================================
# S1 Visualization: ORA Hallmark — Dot Plot + Heatmap
# ==============================================================================
# Input:   ora_bulk_degs_all.csv
# Output:  ora_hallmark_dotplot.png/.pdf
#          ora_hallmark_heatmap.png/.pdf
# ==============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)

# --- Configuration ---
input_file <- "D:/HGSOC_Spatial_Atlas/05_analysis_ready/bulk_ora/ora_bulk_degs_all.csv"
output_dir <- "D:/HGSOC_Spatial_Atlas/05_analysis_ready/bulk_ora"

# --- Load data ---
df <- read.csv(input_file, sep = ";", stringsAsFactors = FALSE)
hall <- df[df$database == "Hallmark", ]

# Clean pathway names
hall$pathway <- gsub("HALLMARK_", "", hall$ID)
hall$pathway <- gsub("_", " ", hall$pathway)

# Signed -log10(padj): positive for UP, negative for DOWN
hall$neg_log10_padj <- -log10(hall$p.adjust)
hall$signed_score <- ifelse(hall$direction == "UP",
                            hall$neg_log10_padj,
                            -hall$neg_log10_padj)

# Cap extreme values for display (EMT is ~29, everything else < 18)
CAP <- 10
hall$signed_score_capped <- pmax(pmin(hall$signed_score, CAP), -CAP)
hall$neg_log10_capped <- pmin(hall$neg_log10_padj, CAP)

# Comparison labels
hall$comp_label <- case_when(
  hall$comparison == "PEO4_SNAI1_vs_GFP"  ~ "SNAI1 vs GFP",
  hall$comparison == "PEO4_2R_vs_GFP"     ~ "2R vs GFP",
  hall$comparison == "PEO4_2R_vs_SNAI1"   ~ "2R vs SNAI1"
)
hall$comp_label <- factor(hall$comp_label,
                          levels = c("SNAI1 vs GFP", "2R vs GFP", "2R vs SNAI1"))

# Combine comparison + direction into a single column for x-axis
hall$comp_dir <- paste0(hall$comp_label, "\n", hall$direction)
hall$comp_dir <- factor(hall$comp_dir,
                        levels = c("SNAI1 vs GFP\nUP", "SNAI1 vs GFP\nDOWN",
                                   "2R vs GFP\nUP", "2R vs GFP\nDOWN",
                                   "2R vs SNAI1\nUP", "2R vs SNAI1\nDOWN"))

# --- Select pathways: significant in at least 1 comparison ---
sig_pathways <- unique(hall$pathway[hall$p.adjust < 0.05])

hall_plot <- hall[hall$pathway %in% sig_pathways, ]

# Sort pathways: by max signed score in SNAI1 vs GFP (to group UP/DOWN)
pathway_order <- hall_plot %>%
  filter(comp_label == "SNAI1 vs GFP") %>%
  group_by(pathway) %>%
  summarise(best_score = signed_score_capped[which.min(p.adjust)]) %>%
  arrange(best_score) %>%
  pull(pathway)

# Add pathways only sig in other comparisons
extra <- setdiff(sig_pathways, pathway_order)
pathway_order <- c(extra, pathway_order)

hall_plot$pathway <- factor(hall_plot$pathway, levels = pathway_order)

# ==============================================================================
# PLOT 1: Dot plot — comparison x direction on x, pathway on y
# ==============================================================================
p1 <- ggplot(hall_plot, aes(x = comp_dir, y = pathway)) +
  geom_point(aes(fill = signed_score_capped, size = neg_log10_capped),
             shape = 21, color = "black", stroke = 0.3) +
  scale_fill_gradient2(
    low = "#2166AC", mid = "white", high = "#B2182B", midpoint = 0,
    limits = c(-CAP, CAP),
    name = paste0("Signed -log10(padj)\n(capped at +/-", CAP, ")\nRed=UP, Blue=DOWN")
  ) +
  scale_size_continuous(
    range = c(1.5, 9),
    limits = c(0, CAP),
    name = paste0("-log10(padj)\n(capped at ", CAP, ")")
  ) +
  geom_point(data = hall_plot[hall_plot$p.adjust >= 0.05, ],
             aes(x = comp_dir, y = pathway),
             shape = 4, size = 1.5, color = "grey60") +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x = element_text(size = 8, hjust = 0.5),
    axis.text.y = element_text(size = 9),
    panel.grid.major = element_line(color = "grey92"),
    panel.grid.minor = element_blank(),
    legend.position = "right",
    plot.title = element_text(size = 11, face = "bold"),
    plot.subtitle = element_text(size = 9, color = "grey40")
  ) +
  labs(
    title = "S1: Hallmark ORA — PEO4 Bulk DEG Comparisons",
    subtitle = "BH-corrected padj < 0.05, |log2FC| > 1 | Grey X = not significant",
    x = NULL, y = NULL
  )

ggsave(file.path(output_dir, "ora_hallmark_dotplot.png"), p1,
       width = 10, height = 6, dpi = 300)
ggsave(file.path(output_dir, "ora_hallmark_dotplot.pdf"), p1,
       width = 10, height = 6)
cat("Saved: ora_hallmark_dotplot\n")

# ==============================================================================
# PLOT 2: Heatmap — cleaner view, one cell per comparison x pathway
# ==============================================================================
# For heatmap: use the BEST (most significant) result per pathway x comparison
# regardless of direction, but encode direction in the sign

hall_heatmap <- hall_plot %>%
  group_by(pathway, comp_label) %>%
  slice_min(p.adjust, n = 1) %>%
  ungroup() %>%
  select(pathway, comp_label, signed_score_capped, p.adjust, direction)

# Complete the grid (fill missing combos with NA)
grid <- expand.grid(
  pathway = levels(hall_plot$pathway),
  comp_label = levels(hall$comp_label),
  stringsAsFactors = FALSE
)
hall_hm <- left_join(grid, hall_heatmap, by = c("pathway", "comp_label"))
hall_hm$pathway <- factor(hall_hm$pathway, levels = pathway_order)

# Significance label
hall_hm$sig_label <- case_when(
  is.na(hall_hm$p.adjust)  ~ "",
  hall_hm$p.adjust < 0.001 ~ "***",
  hall_hm$p.adjust < 0.01  ~ "**",
  hall_hm$p.adjust < 0.05  ~ "*",
  TRUE                     ~ ""
)

# Direction arrow
hall_hm$dir_label <- case_when(
  is.na(hall_hm$direction) ~ "",
  hall_hm$p.adjust >= 0.05 ~ "",
  hall_hm$direction == "UP" ~ paste0("\u2191", hall_hm$sig_label),
  hall_hm$direction == "DOWN" ~ paste0("\u2193", hall_hm$sig_label)
)

p2 <- ggplot(hall_hm, aes(x = comp_label, y = pathway)) +
  geom_tile(aes(fill = signed_score_capped), color = "white", linewidth = 0.8) +
  geom_text(aes(label = dir_label), size = 3.5, fontface = "bold") +
  scale_fill_gradient2(
    low = "#2166AC", mid = "grey95", high = "#B2182B", midpoint = 0,
    limits = c(-CAP, CAP),
    na.value = "grey85",
    name = paste0("Signed -log10(padj)\n(capped +/-", CAP, ")")
  ) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x = element_text(size = 10, face = "bold"),
    axis.text.y = element_text(size = 9),
    panel.grid = element_blank(),
    legend.position = "right",
    plot.title = element_text(size = 11, face = "bold"),
    plot.subtitle = element_text(size = 9, color = "grey40")
  ) +
  labs(
    title = "S1: Hallmark Pathway Enrichment — Bulk PEO4 Comparisons",
    subtitle = expression(paste(
      "Arrow = direction (", uparrow, "UP / ", downarrow, "DOWN) | ",
      "* padj<0.05  ** padj<0.01  *** padj<0.001"
    )),
    x = NULL, y = NULL
  )

ggsave(file.path(output_dir, "ora_hallmark_heatmap.png"), p2,
       width = 8, height = 6, dpi = 300)
ggsave(file.path(output_dir, "ora_hallmark_heatmap.pdf"), p2,
       width = 8, height = 6)
cat("Saved: ora_hallmark_heatmap\n")

cat("\n=== DONE ===\n")
