

metric_df <- readRDS("data/metric_df.rds")
hamming_summary <- readRDS("data/hamming_summary.rds")

# ========================================
# Plots
# ========================================

# Hamming Loss Plot
ggplot(hamming_summary, aes(x = Embedding, y = Hamming_Loss, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  labs(
    title = "Hamming Loss by Model and Embedding",
    y = "Hamming Loss",
    x = "Embedding"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
    axis.title.x = element_text(face = "bold", hjust = 0.5),
    axis.title.y = element_text(face = "bold", hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# Genre-wise Plot
metric_long <- metric_df %>%
  pivot_longer(cols = c(Precision, Recall, F1), names_to = "Metric", values_to = "Score")

ggplot(metric_long, aes(x = Embedding, y = Score, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_grid(Genre ~ Model) +
  theme_minimal() +
  labs(title = "Genre-wise Precision, Recall, F1 by Model and Embedding", y = "Score", x = "Embedding") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Filtering for F-1 scores
f1_long <- metric_df %>%
  pivot_longer(cols = c(Precision, Recall, F1), names_to = "Metric", values_to = "Score") %>%
  filter(Metric == "F1")

# F1 Score Plot
ggplot(f1_long, aes(x = Embedding, y = Score, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", color = "black") +
  facet_wrap(~ Genre) +
  theme_minimal() +
  labs(
    title = "Genre-wise F1 Score by Model and Embedding",
    y = "F1 Score",
    x = "Embedding"
  ) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
    axis.title.x = element_text(face = "bold", hjust = 0.5),
    axis.title.y = element_text(face = "bold", hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

