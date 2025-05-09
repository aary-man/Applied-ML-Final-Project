
df_model <- readRDS("data/df_model.rds")


# ========================================
# Embeddings
# ========================================

# ========================================
# Bag-of-Words 
# ========================================

stop_words <- stopwords("en")

preprocess_str <- function(text) {
  text <- tolower(text)
  text <- gsub("[^a-z\\s]", " ", text)          
  text <- gsub("\\s+", " ", text)               
  text <- trimws(text)                          
  tokens <- unlist(strsplit(text, "\\s+"))      
  tokens <- tokens[!(tokens %in% stop_words)]   
  tokens <- lemmatize_words(tokens)             
  paste(tokens, collapse = " ")                 
}

X_pre_bow <- sapply(df_model$Plot, preprocess_str)

# example
cat("Original Plot:\n", df_model$Plot[1], "\n")
cat("Preprocessed Plot:\n", X_pre_bow[1], "\n")

# comparison of original and preprocessed text
head(data.frame(Original = df_model$Plot, Preprocessed = X_pre_bow))

# ========================================
# BoW
# ========================================
#test and train split
y <- df_model[, selected_genres]
set.seed(42)
train_index <- createDataPartition(1:nrow(df_model), p = 0.8, list = FALSE)

X_train_raw <- X_pre_bow[train_index]
X_test_raw  <- X_pre_bow[-train_index]
y_train <- y[train_index, ]
y_test  <- y[-train_index, ]

saveRDS(y_train, "data/y_train.rds")
saveRDS(y_test, "data/y_test.rds")

# ========================================
# Bag-of-Words 
# ========================================

it_train <- itoken(X_train_raw, tokenizer = word_tokenizer, progressbar = FALSE)
it_test  <- itoken(X_test_raw, tokenizer = word_tokenizer, progressbar = FALSE)

vocab <- create_vocabulary(it_train)
vocab <- prune_vocabulary(vocab, vocab_term_max = 5000)
vectorizer <- vocab_vectorizer(vocab)

X_train_bow <- create_dtm(it_train, vectorizer)
X_test_bow  <- create_dtm(it_test, vectorizer)

saveRDS(X_train_bow, "data/X_train_bow.rds")
saveRDS(X_test_bow, "data/X_test_bow.rds")

# checking dimensions
cat("X_train_bow shape:", dim(X_train_bow), "\n")
cat("X_test_bow shape:", dim(X_test_bow), "\n")
cat("y_train shape:", dim(y_train), "\n")
cat("y_test shape:", dim(y_test), "\n")


# ========================================
# TF-IDF 
# ========================================

tfidf_transformer <- TfIdf$new()

X_train_tfidf <- tfidf_transformer$fit_transform(X_train_bow)
X_test_tfidf  <- tfidf_transformer$transform(X_test_bow)

saveRDS(X_train_tfidf, "data/X_train_tfidf.rds")
saveRDS(X_test_tfidf, "data/X_test_tfidf.rds")

# shape of transformed matrices
cat("X_train_tfidf shape:", dim(X_train_tfidf), "\n")
cat("X_test_tfidf shape:", dim(X_test_tfidf), "\n")


# ========================================
# Word2Vec
# ========================================

# tokenizing
tokens <- word_tokenizer(X_pre_bow)

# rows w 0 genres
df_all_genres_zero <- df_model[rowSums(df_model[, selected_genres]) == 0, ]
tokens_zero <- word_tokenizer(df_all_genres_zero$Plot)

tokens_train <- tokens[train_index]
tokens_w2v_training <- c(tokens_train, tokens_zero)

train_texts <- sapply(tokens_w2v_training, paste, collapse = " ")

# training Word2Vec (CBOW)
w2v_model <- word2vec(train_texts, type = "cbow", dim = 100, window = 5, iter = 5, min_count = 1)

# extracting vectors
embedding_matrix <- as.matrix(w2v_model)

get_doc_vector <- function(doc, embedding_matrix) {
  valid_words <- intersect(doc, rownames(embedding_matrix))
  if (length(valid_words) == 0) return(rep(0, ncol(embedding_matrix)))
  colMeans(embedding_matrix[valid_words, , drop = FALSE])
}

X_train_w2v <- t(sapply(tokens[train_index], get_doc_vector, embedding_matrix))
X_test_w2v  <- t(sapply(tokens[-train_index], get_doc_vector, embedding_matrix))

saveRDS(X_train_w2v, "data/X_train_w2v.rds")
saveRDS(X_test_w2v, "data/X_test_w2v.rds")

# checking shapes
cat("X_train_w2v shape:", dim(X_train_w2v), "\n")
cat("X_test_w2v shape:", dim(X_test_w2v), "\n")
cat("y_train shape:", dim(y_train), "\n")
cat("y_test shape:", dim(y_test), "\n")


# ========================================
# GloVe Embeddings (50D)
# ========================================

# 1. loading GloVe vectors. Downloaded from internet
glove_path <- "glove.6B.50d.txt"  

cat("Loading GloVe 50d vectors...\n")
glove_lines <- readLines(glove_path)
glove_split <- strsplit(glove_lines, " ")

# extracting words and vectors
words <- sapply(glove_split, `[[`, 1)
vectors <- t(sapply(glove_split, function(x) as.numeric(x[-1])))
rownames(vectors) <- words
embedding_matrix <- vectors
rm(glove_lines, glove_split, words, vectors)

cat("Loaded GloVe with", nrow(embedding_matrix), "words and", ncol(embedding_matrix), "dimensions\n")

# processing text into tokenized lists
tokens <- word_tokenizer(X_pre_bow)

glove_document_embedding <- function(words, embedding_matrix) {
  valid_words <- intersect(words, rownames(embedding_matrix))
  if (length(valid_words) == 0) return(rep(0, ncol(embedding_matrix)))
  colMeans(embedding_matrix[valid_words, , drop = FALSE])
}

# applying to train/test
X_train_glove <- t(sapply(tokens[train_index], glove_document_embedding, embedding_matrix))
X_test_glove  <- t(sapply(tokens[-train_index], glove_document_embedding, embedding_matrix))

saveRDS(X_train_glove, "data/X_train_glove.rds")
saveRDS(X_test_glove, "data/X_test_glove.rds")

# checking shapes
cat("X_train_glove shape:", dim(X_train_glove), "\n")
cat("X_test_glove shape:", dim(X_test_glove), "\n")
cat("y_train shape:", dim(y_train), "\n")
cat("y_test shape:", dim(y_test), "\n")


