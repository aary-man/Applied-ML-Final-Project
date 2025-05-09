X_train_tfidf <- readRDS("data/X_train_tfidf.rds")
X_test_tfidf <- readRDS("data/X_test_tfidf.rds")
y_train <- readRDS("data/y_train.rds")
y_test <- readRDS("data/y_test.rds")


# ========================================
# KMeans Clustering on TF-IDF
# ========================================


# Combining TF-IDF embeddings and true labels
X_all <- rbind(as.matrix(X_train_tfidf), as.matrix(X_test_tfidf))
y_all <- rbind(y_train, y_test)

# elbow plot to find best k
elbow_plot <- fviz_nbclust(X_all, kmeans, method = "wss", k.max = 12) +
  labs(title = "Elbow Plot for KMeans (TF-IDF)", x = "Number of Clusters", y = "WSS")+
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
    axis.title.x = element_text(face = "bold", hjust = 0.5),
    axis.title.y = element_text(face = "bold", hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# running k- means
k <- 6
set.seed(42)
kmeans_result <- kmeans(X_all, centers = k, nstart = 25)
clusters <- kmeans_result$cluster
centroids <- kmeans_result$centers

# comparing clusters to true labels
y_clustered <- as.data.frame(y_all)
y_clustered$Cluster <- as.factor(clusters)

cluster_genre_summary <- y_clustered %>%
  group_by(Cluster) %>%
  summarise(across(where(is.numeric), mean))

cluster_genre_long <- cluster_genre_summary %>%
  pivot_longer(-Cluster, names_to = "Genre", values_to = "Proportion")

cluster_plot <- ggplot(cluster_genre_long, aes(x = Genre, y = Proportion, fill = Cluster)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Genre Composition per Cluster (KMeans on TF-IDF)", x = "Genre", y = "Proportion") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))+
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5, size = 14),
    axis.title.x = element_text(face = "bold", hjust = 0.5),
    axis.title.y = element_text(face = "bold", hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# distance to centroids
dist_mat <- as.matrix(proxy::dist(X_all, centroids, method = "euclidean"))
min_dist <- apply(dist_mat, 1, min)
second_min <- apply(dist_mat, 1, function(x) sort(x)[2])
delta <- second_min - min_dist

# books near multiple centroids with 25% threshold
threshold <- quantile(delta, 0.25)  
predicted_mixed <- delta < threshold
true_mixed <- rowSums(y_all == 1) > 1

contingency_table <- table(TrueMulti = true_mixed, PredictedMixed = predicted_mixed)

# comparing kmeans to svm linear + tf-idf
tp <- 405
fp <- 1303
fn <- 1423

precision_kmeans <- tp / (tp + fp)       
recall_kmeans <- tp / (tp + fn)          
f1_kmeans <- 2 * precision_kmeans * recall_kmeans / (precision_kmeans + recall_kmeans)

# svm linear+ tf-idf metrics
precision_supervised <- 0.692
recall_supervised <- 0.741
f1_supervised <- 0.716

# comparison
top_model_metrics <- tibble(
  Model = c("KMeans (TF-IDF)", "SVM Linear (TF-IDF)"),
  Precision = c(round(precision_kmeans, 3), precision_supervised),
  Recall = c(round(recall_kmeans, 3), recall_supervised),
  F1 = c(round(f1_kmeans, 3), f1_supervised)
)

print(top_model_metrics)

# qualitative comparison of 10 books
set.seed(1244)

# Sampling
case_comparison <- y_clustered %>%
  mutate(BookID = row_number()) %>%
  group_by(Cluster) %>%
  slice_sample(n = 2, replace = FALSE) %>%
  ungroup() %>%
  slice_head(n = 10) %>%  
  mutate(
    TrueGenres = apply(select(., -BookID, -Cluster), 1, function(x) paste(names(x)[x == 1], collapse = ", ")),
    ClusterGenres = sapply(as.integer(Cluster), function(cl) {
      top_genres <- sort(unlist(cluster_genre_summary[cl, -1]), decreasing = TRUE)
      paste(names(top_genres)[1:2], collapse = ", ")
    })
  ) %>%
  select(BookID, Cluster, TrueGenres, ClusterGenres)

# all plots and results stored
stored_results <- list(
  elbow_plot = elbow_plot,
  cluster_plot = cluster_plot,
  cluster_summary = cluster_genre_summary,
  contingency_table = contingency_table,
  model_metrics = top_model_metrics,
  case_comparison = case_comparison
)


saveRDS(cluster_plot, "data/cluster_plot.rds")
saveRDS(elbow_plot, "data/elbow_plot.rds")
