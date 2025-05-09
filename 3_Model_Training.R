

X_train_bow <- readRDS("data/X_train_bow.rds")
X_test_bow <- readRDS("data/X_test_bow.rds")

X_train_tfidf <- readRDS("data/X_train_tfidf.rds")
X_test_tfidf <- readRDS("data/X_test_tfidf.rds")

X_train_w2v <- readRDS("data/X_train_w2v.rds")
X_test_w2v <- readRDS("data/X_test_w2v.rds")

X_train_glove <- readRDS("data/X_train_glove.rds")
X_test_glove <- readRDS("data/X_test_glove.rds")

y_train <- readRDS("data/y_train.rds")
y_test <- readRDS("data/y_test.rds")



# ========================================
# Model Training
# ========================================



X_by_embed <- list(
  "Bag of Words" = list(train = X_train_bow, test = X_test_bow),
  "TF-IDF" = list(train = X_train_tfidf, test = X_test_tfidf),
  "Word2Vec" = list(train = X_train_w2v, test = X_test_w2v),
  "GloVe" = list(train = X_train_glove, test = X_test_glove)
)

hamming_loss <- function(true, pred) {
  mean(true != pred)
}

metric_df <- tibble()
hamming_summary <- tibble()
threshold <- 0.5

model_names <- c("Logistic Regression", "SVM (Linear)", "SVM (RBF)", "Decision Tree", "Random Forest")

# Logistic Regression
for (embed_name in names(X_by_embed)) {
  X_train <- as.matrix(X_by_embed[[embed_name]]$train)
  X_test  <- as.matrix(X_by_embed[[embed_name]]$test)
  y_pred_bin <- matrix(0, nrow = nrow(X_test), ncol = ncol(y_test))
  colnames(y_pred_bin) <- colnames(y_test)
  for (i in seq_len(ncol(y_train))) {
    y_col <- as.numeric(y_train[[i]])
    fit <- cv.glmnet(X_train, y_col, family = "binomial", alpha = 0)
    preds <- predict(fit, newx = X_test, s = "lambda.min", type = "response")
    y_pred_bin[, i] <- ifelse(preds > threshold, 1, 0)
  }
  loss <- hamming_loss(as.matrix(y_test), y_pred_bin)
  hamming_summary <- bind_rows(hamming_summary, tibble(Model = "Logistic Regression", Embedding = embed_name, Hamming_Loss = loss))
  for (i in seq_len(ncol(y_test))) {
    genre <- colnames(y_test)[i]
    truth <- factor(y_test[[i]], levels = c(0, 1))
    prediction <- factor(y_pred_bin[, i], levels = c(0, 1))
    m <- metric_set(precision, recall, f_meas)(data.frame(truth = truth, prediction = prediction), truth = truth, estimate = prediction)
    metric_df <- bind_rows(metric_df, tibble(Model = "Logistic Regression", Embedding = embed_name, Genre = genre,
                                             Precision = m$.estimate[m$.metric == "precision"],
                                             Recall = m$.estimate[m$.metric == "recall"],
                                             F1 = m$.estimate[m$.metric == "f_meas"]))
  }
}

# SVM (Linear and RBF)
kernels <- c("linear", "radial")
for (kernel in kernels) {
  for (embed_name in names(X_by_embed)) {
    model_label <- paste0("SVM (", ifelse(kernel == "linear", "Linear", "RBF"), ")")
    X_train <- as.matrix(X_by_embed[[embed_name]]$train)
    X_test <- as.matrix(X_by_embed[[embed_name]]$test)
    y_pred_bin <- matrix(0, nrow = nrow(X_test), ncol = ncol(y_test))
    colnames(y_pred_bin) <- colnames(y_test)
    for (i in seq_len(ncol(y_train))) {
      model <- svm(x = X_train, y = as.factor(y_train[[i]]), kernel = kernel, cost = 1, scale = FALSE, probability = TRUE)
      prob <- attr(predict(model, X_test, probability = TRUE), "probabilities")[, "1"]
      y_pred_bin[, i] <- ifelse(prob >= 0.5, 1, 0)
    }
    loss <- hamming_loss(as.matrix(y_test), y_pred_bin)
    hamming_summary <- bind_rows(hamming_summary, tibble(Model = model_label, Embedding = embed_name, Hamming_Loss = loss))
    for (i in seq_len(ncol(y_test))) {
      genre <- colnames(y_test)[i]
      truth <- factor(y_test[[i]], levels = c(0, 1))
      prediction <- factor(y_pred_bin[, i], levels = c(0, 1))
      m <- metric_set(precision, recall, f_meas)(data.frame(truth = truth, prediction = prediction), truth = truth, estimate = prediction)
      metric_df <- bind_rows(metric_df, tibble(Model = model_label, Embedding = embed_name, Genre = genre,
                                               Precision = m$.estimate[m$.metric == "precision"],
                                               Recall = m$.estimate[m$.metric == "recall"],
                                               F1 = m$.estimate[m$.metric == "f_meas"]))
    }
  }
}

# Decision Tree
for (embed_name in names(X_by_embed)) {
  X_train <- as.data.frame(as.matrix(X_by_embed[[embed_name]]$train))
  X_test <- as.data.frame(as.matrix(X_by_embed[[embed_name]]$test))
  y_pred_bin <- matrix(0, nrow = nrow(X_test), ncol = ncol(y_test))
  colnames(y_pred_bin) <- colnames(y_test)
  for (i in seq_len(ncol(y_train))) {
    train_df <- cbind(X_train, label = y_train[[i]])
    tree_model <- rpart(label ~ ., data = train_df, method = "class", parms = list(split = "information"), control = rpart.control(maxdepth = 10, cp = 0.01))
    preds <- predict(tree_model, newdata = X_test, type = "class")
    y_pred_bin[, i] <- as.numeric(as.character(preds))
  }
  loss <- hamming_loss(as.matrix(y_test), y_pred_bin)
  hamming_summary <- bind_rows(hamming_summary, tibble(Model = "Decision Tree", Embedding = embed_name, Hamming_Loss = loss))
  for (i in seq_len(ncol(y_test))) {
    genre <- colnames(y_test)[i]
    truth <- factor(y_test[[i]], levels = c(0, 1))
    prediction <- factor(y_pred_bin[, i], levels = c(0, 1))
    m <- metric_set(precision, recall, f_meas)(data.frame(truth = truth, prediction = prediction), truth = truth, estimate = prediction)
    metric_df <- bind_rows(metric_df, tibble(Model = "Decision Tree", Embedding = embed_name, Genre = genre,
                                             Precision = m$.estimate[m$.metric == "precision"],
                                             Recall = m$.estimate[m$.metric == "recall"],
                                             F1 = m$.estimate[m$.metric == "f_meas"]))
  }
}

# Random Forest
for (embed_name in names(X_by_embed)) {
  X_train <- as.data.frame(as.matrix(X_by_embed[[embed_name]]$train))
  X_test <- as.data.frame(as.matrix(X_by_embed[[embed_name]]$test))
  y_pred_bin <- matrix(0, nrow = nrow(X_test), ncol = ncol(y_test))
  colnames(y_pred_bin) <- colnames(y_test)
  for (i in seq_len(ncol(y_train))) {
    rf_model <- randomForest(x = X_train, y = as.factor(y_train[[i]]), ntree = 10, mtry = floor(sqrt(ncol(X_train))), importance = FALSE)
    preds <- predict(rf_model, newdata = X_test, type = "response")
    y_pred_bin[, i] <- as.numeric(as.character(preds))
  }
  loss <- hamming_loss(as.matrix(y_test), y_pred_bin)
  hamming_summary <- bind_rows(hamming_summary, tibble(Model = "Random Forest", Embedding = embed_name, Hamming_Loss = loss))
  for (i in seq_len(ncol(y_test))) {
    genre <- colnames(y_test)[i]
    truth <- factor(y_test[[i]], levels = c(0, 1))
    prediction <- factor(y_pred_bin[, i], levels = c(0, 1))
    m <- metric_set(precision, recall, f_meas)(data.frame(truth = truth, prediction = prediction), truth = truth, estimate = prediction)
    metric_df <- bind_rows(metric_df, tibble(Model = "Random Forest", Embedding = embed_name, Genre = genre,
                                             Precision = m$.estimate[m$.metric == "precision"],
                                             Recall = m$.estimate[m$.metric == "recall"],
                                             F1 = m$.estimate[m$.metric == "f_meas"]))
  }
}


saveRDS(metric_df, "data/metric_df.rds")
saveRDS(hamming_summary, "data/hamming_summary.rds")
