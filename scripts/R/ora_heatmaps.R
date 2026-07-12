# ==============================================================================
# S1 Visualization: Gene-Level Heatmaps
# ==============================================================================
# Heatmap 1: Genes driving significant Hallmark enrichment in SNAI1 vs GFP 
#            and 2R vs GFP, grouped by pathway, showing log2FC across both
# Heatmap 2: The 109 SNAI1-ac signature genes (2R vs SNAI1) with Hallmark
#            pathway annotation
#
# Input:
#   - ora_bulk_degs_all.csv
#   - tt_PEO4-SNAI1-2R_Analysis.xlsx (Cmpr sheet for full FC values)
#   - snai1_acetylation_signature_short.csv (109-gene signature)
#
# Output:
#   - ora_heatmap_vsGFP.png/.pdf
#   - ora_heatmap_2R_vs_SNAI1.png/.pdf
# ==============================================================================

library(readxl)
library(ggplot2)
library(dplyr)
library(tidyr)
library(msigdbr)

# --- Configuration ---
ora_file  <- "D:/HGSOC_Spatial_Atlas/05_analysis_ready/bulk_ora/ora_bulk_degs_all.csv"
xlsx_file <- "C:/Users/luchu/Documents/MSc/Master Thesis/Code/RNA-seq/tt_PEO4-SNAI1-2R_Analysis.xlsx"
sig_file  <- "D:/HGSOC_Spatial_Atlas/05_analysis_ready/Signature/snai1_acetylation_signature_short.csv"
output_dir <- "D:/HGSOC_Spatial_Atlas/05_analysis_ready/bulk_ora"

# ==============================================================================
# Load data
# ==============================================================================
cat("Loading data...\n")

# ORA results
ora <- read.csv(ora_file, sep = ";", stringsAsFactors = FALSE, fileEncoding = "latin1")
hall_sig <- ora[ora$database == "Hallmark" & ora$p.adjust < 0.05, ]
vsGFP <- hall_sig[hall_sig$comparison %in% c("PEO4_SNAI1_vs_GFP", "PEO4_2R_vs_GFP"), ]

# Full expression data (Cmpr sheet) for log2FC values
cmpr <- read_excel(xlsx_file, sheet = "Cmpr")
# Fix character columns
cmpr$`PEO4_SNAI1vsGFP_PPEE`        <- as.numeric(cmpr$`PEO4_SNAI1vsGFP_PPEE`)
cmpr$`PEO4_lg2fc (SNAI1-GFP)`      <- as.numeric(cmpr$`PEO4_lg2fc (SNAI1-GFP)`)
cmpr$`PEO4-2R_SNAI1vsGFP_PPEE`     <- as.numeric(cmpr$`PEO4-2R_SNAI1vsGFP_PPEE`)
cmpr$`PEO4-2R_lg2fc (SNAI1-GFP)`   <- as.numeric(cmpr$`PEO4-2R_lg2fc (SNAI1-GFP)`)
cmpr$`PEO4-2R_SNAI1vsSNAI1_PPEE`   <- as.numeric(cmpr$`PEO4-2R_SNAI1vsSNAI1_PPEE`)
cmpr$`PEO4-2R_lg2fc (SNAI1-SNAI1)` <- as.numeric(cmpr$`PEO4-2R_lg2fc (SNAI1-SNAI1)`)

# 109-gene signature
sig <- read.csv(sig_file, sep = ";", stringsAsFactors = FALSE)

# Hallmark gene sets (for annotating signature genes)
hallmark_df <- msigdbr(species = "Homo sapiens", category = "H")

# ==============================================================================
# HEATMAP 1: vs GFP comparisons — genes grouped by pathway
# ==============================================================================
cat("\n=== Building Heatmap 1: vs GFP ===\n")

# --- Assign each gene to its most significant pathway ---
gene_pathway <- data.frame(gene = character(), pathway = character(),
                           padj = numeric(), stringsAsFactors = FALSE)

for (i in seq_len(nrow(vsGFP))) {
  pathway <- gsub("HALLMARK_", "", vsGFP$ID[i])
  padj_val <- vsGFP$p.adjust[i]
  genes <- strsplit(vsGFP$geneID[i], "/")[[1]]
  for (g in genes) {
    gene_pathway <- rbind(gene_pathway,
                          data.frame(gene = g, pathway = pathway,
                                     padj = padj_val, stringsAsFactors = FALSE))
  }
}

# Keep best (most significant) pathway per gene
gene_pathway <- gene_pathway %>%
  group_by(gene) %>%
  slice_min(padj, n = 1, with_ties = FALSE) %>%
  ungroup()

cat(sprintf("Unique genes: %d across %d pathways\n",
            nrow(gene_pathway), length(unique(gene_pathway$pathway))))

# --- Get log2FC for all 3 comparisons ---
fc_data <- cmpr %>%
  select(Gene,
         SNAI1_vs_GFP = `PEO4_lg2fc (SNAI1-GFP)`,
         `2R_vs_GFP`  = `PEO4-2R_lg2fc (SNAI1-GFP)`,
         `2R_vs_SNAI1` = `PEO4-2R_lg2fc (SNAI1-SNAI1)`) %>%
  filter(Gene %in% gene_pathway$gene)

# Merge with pathway assignment
plot_df <- inner_join(gene_pathway %>% select(gene, pathway), fc_data,
                      by = c("gene" = "Gene"))

cat(sprintf("Genes with FC data: %d\n", nrow(plot_df)))

# --- Order pathways by direction (UP pathways first, then DOWN) ---
# Determine dominant direction per pathway from the ORA results
pathway_direction <- vsGFP %>%
  group_by(ID) %>%
  slice_min(p.adjust, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  mutate(pathway = gsub("HALLMARK_", "", ID)) %>%
  select(pathway, direction, p.adjust)

# Order: UP pathways (by significance), then DOWN pathways (by significance)
up_pathways <- pathway_direction %>%
  filter(direction == "UP") %>%
  arrange(p.adjust) %>%
  pull(pathway)
down_pathways <- pathway_direction %>%
  filter(direction == "DOWN") %>%
  arrange(p.adjust) %>%
  pull(pathway)

pathway_order <- c(up_pathways, down_pathways)
# Add any pathways not captured (edge case)
extra <- setdiff(unique(plot_df$pathway), pathway_order)
pathway_order <- c(pathway_order, extra)

plot_df$pathway <- factor(plot_df$pathway, levels = pathway_order)

# --- Order genes within each pathway by SNAI1 vs GFP FC ---
plot_df <- plot_df %>%
  arrange(pathway, SNAI1_vs_GFP) %>%
  mutate(gene = factor(gene, levels = unique(gene)))

# --- Pivot to long format for ggplot ---
plot_long <- plot_df %>%
  pivot_longer(cols = c(SNAI1_vs_GFP, `2R_vs_GFP`, `2R_vs_SNAI1`),
               names_to = "comparison", values_to = "log2FC")

plot_long$comparison <- factor(plot_long$comparison,
                               levels = c("SNAI1_vs_GFP", "2R_vs_GFP", "2R_vs_SNAI1"))

# --- Cap FC for display ---
FC_CAP <- 6
plot_long$log2FC_capped <- pmax(pmin(plot_long$log2FC, FC_CAP), -FC_CAP)

# --- Plot ---
p1 <- ggplot(plot_long, aes(x = comparison, y = gene, fill = log2FC_capped)) +
  geom_tile(color = "white", linewidth = 0.1) +
  scale_fill_gradient2(
    low = "#2166AC", mid = "white", high = "#B2182B", midpoint = 0,
    limits = c(-FC_CAP, FC_CAP),
    name = paste0("log2FC\n(capped +/-", FC_CAP, ")")
  ) +
  facet_grid(pathway ~ ., scales = "free_y", space = "free_y", switch = "y") +
  scale_x_discrete(labels = c("SNAI1 vs GFP", "2R vs GFP", "2R vs SNAI1")) +
  theme_minimal(base_size = 8) +
  theme(
    axis.text.y = element_text(size = 5),
    axis.text.x = element_text(size = 9, angle = 45, hjust = 1),
    strip.text.y.left = element_text(size = 7, angle = 0, hjust = 1, face = "bold"),
    strip.placement = "outside",
    panel.spacing = unit(0.5, "lines"),
    panel.grid = element_blank(),
    legend.position = "right",
    plot.title = element_text(size = 11, face = "bold"),
    plot.subtitle = element_text(size = 8, color = "grey40")
  ) +
  labs(
    title = "S1: Gene-Level log2FC — SNAI1 Overexpression (vs GFP)",
    subtitle = "Genes grouped by most significant Hallmark pathway | DEGs: padj<0.05, |log2FC|>1",
    x = NULL, y = NULL
  )

# Dynamic height based on gene count
fig_height <- max(12, nrow(plot_df) * 0.08 + 3)

ggsave(file.path(output_dir, "ora_heatmap_vsGFP.png"), p1,
       width = 7, height = fig_height, dpi = 300, limitsize = FALSE)
ggsave(file.path(output_dir, "ora_heatmap_vsGFP.pdf"), p1,
       width = 7, height = fig_height, limitsize = FALSE)
cat(sprintf("Saved: ora_heatmap_vsGFP (%d genes, height=%.1f)\n",
            nrow(plot_df), fig_height))

# ==============================================================================
# HEATMAP 2: 2R vs SNAI1 — 109 signature genes with Hallmark annotation
# ==============================================================================
cat("\n=== Building Heatmap 2: 2R vs SNAI1 (109 signature genes) ===\n")

# --- Annotate signature genes with Hallmark memberships ---
sig_genes <- sig$Gene

# Map gene to Hallmark pathways
hallmark_map <- hallmark_df %>%
  filter(gene_symbol %in% sig_genes) %>%
  select(gene_symbol, gs_name) %>%
  mutate(gs_name = gsub("HALLMARK_", "", gs_name)) %>%
  distinct()

# For each gene, pick most common pathway or "No Hallmark" if none
sig_pathway <- hallmark_map %>%
  group_by(gene_symbol) %>%
  # If gene is in multiple pathways, pick one (alphabetically first for consistency)
  slice_min(gs_name, n = 1) %>%
  ungroup() %>%
  rename(gene = gene_symbol, pathway = gs_name)

# Add genes with no Hallmark membership
no_hallmark <- setdiff(sig_genes, sig_pathway$gene)
if (length(no_hallmark) > 0) {
  sig_pathway <- rbind(sig_pathway,
                       data.frame(gene = no_hallmark, pathway = "No Hallmark",
                                  stringsAsFactors = FALSE))
}

cat(sprintf("Signature genes mapped to Hallmark: %d / %d\n",
            sum(sig_pathway$pathway != "No Hallmark"), length(sig_genes)))

# --- Get log2FC for all 3 comparisons ---
fc_sig <- cmpr %>%
  select(Gene,
         SNAI1_vs_GFP = `PEO4_lg2fc (SNAI1-GFP)`,
         `2R_vs_GFP`  = `PEO4-2R_lg2fc (SNAI1-GFP)`,
         `2R_vs_SNAI1` = `PEO4-2R_lg2fc (SNAI1-SNAI1)`) %>%
  filter(Gene %in% sig_genes)

plot_sig <- inner_join(sig_pathway, fc_sig, by = c("gene" = "Gene"))

# --- Order: group by pathway, within pathway by 2R vs SNAI1 FC ---
# Put "No Hallmark" last
pathways_present <- sort(unique(plot_sig$pathway[plot_sig$pathway != "No Hallmark"]))
pathway_order2 <- c(pathways_present, "No Hallmark")

plot_sig$pathway <- factor(plot_sig$pathway, levels = pathway_order2)
plot_sig <- plot_sig %>%
  arrange(pathway, `2R_vs_SNAI1`) %>%
  mutate(gene = factor(gene, levels = unique(gene)))

# --- Pivot long ---
plot_sig_long <- plot_sig %>%
  pivot_longer(cols = c(SNAI1_vs_GFP, `2R_vs_GFP`, `2R_vs_SNAI1`),
               names_to = "comparison", values_to = "log2FC")

plot_sig_long$comparison <- factor(plot_sig_long$comparison,
                                   levels = c("SNAI1_vs_GFP", "2R_vs_GFP", "2R_vs_SNAI1"))

plot_sig_long$log2FC_capped <- pmax(pmin(plot_sig_long$log2FC, FC_CAP), -FC_CAP)

# --- Plot ---
p2 <- ggplot(plot_sig_long, aes(x = comparison, y = gene, fill = log2FC_capped)) +
  geom_tile(color = "white", linewidth = 0.1) +
  scale_fill_gradient2(
    low = "#2166AC", mid = "white", high = "#B2182B", midpoint = 0,
    limits = c(-FC_CAP, FC_CAP),
    name = paste0("log2FC\n(capped +/-", FC_CAP, ")")
  ) +
  facet_grid(pathway ~ ., scales = "free_y", space = "free_y", switch = "y") +
  scale_x_discrete(labels = c("SNAI1 vs GFP", "2R vs GFP", "2R vs SNAI1")) +
  theme_minimal(base_size = 8) +
  theme(
    axis.text.y = element_text(size = 5),
    axis.text.x = element_text(size = 9, angle = 45, hjust = 1),
    strip.text.y.left = element_text(size = 6, angle = 0, hjust = 1, face = "bold"),
    strip.placement = "outside",
    panel.spacing = unit(0.5, "lines"),
    panel.grid = element_blank(),
    legend.position = "right",
    plot.title = element_text(size = 11, face = "bold"),
    plot.subtitle = element_text(size = 8, color = "grey40")
  ) +
  labs(
    title = "S1: 109 SNAI1-ac Signature Genes — log2FC Across Comparisons",
    subtitle = "Annotated by Hallmark pathway membership | Sorted by 2R vs SNAI1 FC",
    x = NULL, y = NULL
  )

fig_height2 <- max(10, nrow(plot_sig) * 0.09 + 3)

ggsave(file.path(output_dir, "ora_heatmap_2R_vs_SNAI1.png"), p2,
       width = 7, height = fig_height2, dpi = 300, limitsize = FALSE)
ggsave(file.path(output_dir, "ora_heatmap_2R_vs_SNAI1.pdf"), p2,
       width = 7, height = fig_height2, limitsize = FALSE)
cat(sprintf("Saved: ora_heatmap_2R_vs_SNAI1 (%d genes, height=%.1f)\n",
            nrow(plot_sig), fig_height2))

cat("\n=== DONE ===\n")