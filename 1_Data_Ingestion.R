
# Final Project - Applied Machine Learning
# ========================================

library(dplyr)
library(stringr)
library(tm)
library(textstem)
library(jsonlite)
library(caret)
library(text2vec)
library(randomForest)
library(e1071)
library(class)
library(Matrix)
library(word2vec)
library(utiml)
library(mldr)
library(doc2vec)
library(tokenizers)
library(glmnet)
library(yardstick)
library(kernlab)
library(rpart)
library(e1071)
library(factoextra)
library(proxy)
library(dplyr)
library(ggplot2)
library(tidyr)
library(reshape2)
library(glmnet)
library(e1071)
library(rpart)
library(randomForest)
library(yardstick)
library(dplyr)
library(tidyr)
library(ggplot2)

# ========================================
# Data
# ========================================

df <- read.delim("booksummaries.txt", header = FALSE, sep = "\t", quote = "", stringsAsFactors = FALSE)
colnames(df) <- c("Wiki_ID", "Freebase_ID", "Title", "Author", "Pub_Date", "Genres", "Plot")
cat("Dataset loaded. Rows:", nrow(df), "\n")
head(df)

# droppinh NAs
df <- df[!is.na(df$Genres) & !is.na(df$Plot), ]
cat("After dropping NAs. Rows:", nrow(df), "\n")

# ========================================
# Cleaning
# ========================================
# cleaning summaries
stop_words <- stopwords("en")
clean_text <- function(text) {
  text <- tolower(text)
  text <- removePunctuation(text)
  text <- removeNumbers(text)
  text <- stripWhitespace(text)
  text <- removeWords(text, stop_words)
  lemmatize_strings(text)
}
df$Clean_Plot <- sapply(df$Plot, clean_text)

# ========================================
# Genre Parsing
# ========================================
# parsing genres
parse_genres <- function(x) {
  tryCatch({
    genres <- fromJSON(x)
    if (length(genres) == 0) return(NA)
    paste(unlist(genres), collapse = ", ")
  }, error = function(e) NA)
}
df$Parsed_Genres <- sapply(df$Genres, parse_genres)

# dropping Nas
df <- df[!is.na(df$Parsed_Genres), ]
cat("After genre parsing. Rows:", nrow(df), "\n")

# genre binarizng
genre_list <- strsplit(df$Parsed_Genres, ", ")
all_genres <- sort(unique(unlist(genre_list)))
genre_matrix <- matrix(0, nrow = length(genre_list), ncol = length(all_genres))
colnames(genre_matrix) <- all_genres
for (i in seq_along(genre_list)) {
  genre_matrix[i, genre_list[[i]]] <- 1
}
genre_df <- as.data.frame(genre_matrix)

selected_genres <- c("Science Fiction", "Fantasy", "Mystery", "Suspense",
                     "Crime Fiction", "Thriller", "Horror")
selected_genres <- intersect(selected_genres, colnames(genre_df))
genre_df <- genre_df[, selected_genres]
cat("Genres kept:", paste(selected_genres, collapse = ", "), "\n")

df_model <- cbind(df, genre_df)

df_all_genres_zero <- df_model[rowSums(df_model[, selected_genres]) == 0, ]

df_model <- df_model[rowSums(df_model[, selected_genres]) > 0, ]

saveRDS(df_model, file = "data/df_model.rds")

# ========================================
# Genre Distribution Bar Plot
# ========================================

genre_counts <- colSums(df_model[, selected_genres])
genre_bar_df <- data.frame(Genre = names(genre_counts), Count = as.numeric(genre_counts))

ggplot(genre_bar_df, aes(x = reorder(Genre, -Count), y = Count, fill = Genre)) +
  geom_bar(stat = "identity", color = "black") +
  labs(title = "Genre Distribution in Book Summary Dataset", x = "Genre", y = "Number of Books") +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    axis.title = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  )


# ========================================
# Summary Statistics Before and After Cleaning
# ========================================

# before
df_raw <- read.delim("booksummaries.txt", header = FALSE, sep = "\t", quote = "", stringsAsFactors = FALSE)
colnames(df_raw) <- c("Wiki_ID", "Freebase_ID", "Title", "Author", "Pub_Date", "Genres", "Plot")

raw_stats <- data.frame(
  Stage = "Raw",
  Total_Records = nrow(df_raw),
  NonMissing_Plots = sum(!is.na(df_raw$Plot)),
  NonMissing_Genres = sum(!is.na(df_raw$Genres)),
  Unique_Authors = length(unique(df_raw$Author)),
  Pub_Years_Available = sum(!is.na(as.numeric(df_raw$Pub_Date))),
  Median_Plot_Length = median(nchar(na.omit(df_raw$Plot)), na.rm = TRUE)
)

# after
clean_stats <- data.frame(
  Stage = "Cleaned",
  Total_Records = nrow(df_model),
  NonMissing_Plots = sum(!is.na(df_model$Plot)),
  NonMissing_Genres = sum(!is.na(df_model$Genres)),
  Unique_Authors = length(unique(df_model$Author)),
  Pub_Years_Available = sum(!is.na(as.numeric(df_model$Pub_Date))),
  Median_Plot_Length = median(nchar(df_model$Plot), na.rm = TRUE)
)


summary_comparison <- rbind(raw_stats, clean_stats)

print(summary_comparison)




