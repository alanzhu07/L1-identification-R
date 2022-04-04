# Library Imports
library(readtext)
library(readxl)
library(tidyverse)
library(quanteda)
library(quanteda.extras)
library(udpipe)
library(dendextend)
library(factoextra)
library(glmnet)
library(randomForest)
library(caret)


# Reading Corpus
files_list <- list.files("bawe/CORPUS_TXT",
                         full.names = T, pattern = "*.txt", recursive = T)
bawe_text <- readtext(files_list)
bawe_corpus <- corpus(bawe_text)
bawe_tkns <- tokens(bawe_corpus, what="word", remove_punct=T, remove_symbols=T,
                    remove_numbers=T, remove_url=T)
bawe_dfm <- dfm(bawe_tkns)


# Reading Meta File
bawe.meta <- read_excel("bawe/documentation/BAWE.xls")
names(bawe.meta) <- make.names(names(bawe.meta))


# Corpus Summary
l1.summary <- bawe.meta %>% 
  group_by(L1) %>% 
  dplyr::summarize(count = n()) %>% 
  arrange(desc(count)) %>% 
  mutate(proportion = count/sum(count)*100)
bawe.summary <- data.frame(files = ndoc(bawe_corpus),
                           tokens = sum(ntoken(bawe_tkns))) %>% 
  mutate(token.p.file = tokens/files)


# POS Tagging and creating DFM
ud_model <- udpipe_load_model("english-ewt-ud-2.5-191206.udpipe")
annotation <- udpipe_annotate(ud_model, x = bawe_text$text,
                              doc_id = bawe_text$doc_id, parser = "none")
anno_edit <- annotation %>%
  as_tibble() %>%
  unite("upos", upos:xpos)
sub_tokens <- split(anno_edit$upos, anno_edit$doc_id)
sub_tokens <- as.tokens(sub_tokens)
sub_tokens <- tokens_remove(sub_tokens, "^punct_\\S+", valuetype = "regex")
sub_tokens <- tokens_remove(sub_tokens, "^sym_\\S+", valuetype = "regex")
sub_tokens <- tokens_remove(sub_tokens, "^x_\\S+", valuetype = "regex")
sub_dfm <- sub_tokens %>%
  dfm() %>%
  dfm_weight(scheme = "prop") %>%
  convert(to = "data.frame")
sub_dfm <- sub_dfm %>%
  column_to_rownames("doc_id") %>%
  dplyr::select(order(colnames(.)))
sub_dfm <- sub_dfm %>%
  scale() %>%
  data.frame()


# Hierarchical Agglomerative Clustering
set.seed(42)
l1s <- c("English", "Chinese Cantonese", "Japanese", "French", "German")
bawe_l1 <- bawe.meta %>%
  filter(L1 %in% l1s) %>% 
  group_by(L1) %>% 
  sample_n(10) %>% 
  arrange(id)
sub_dfm_l1 <- sub_dfm[bawe_l1$id,]
zero_var_cols <- which(apply(sub_dfm_l1, 2, var)==0)
sub_dfm_new <- sub_dfm_l1[,-zero_var_cols]
l1_abbr <- plyr::mapvalues(bawe_l1$L1, l1s, c("E", "C", "J", "F", "G"))
l1_colors <- plyr::mapvalues(bawe_l1$L1, l1s, c("black", "orange", "red", "blue", "grey"))
dist.mat <- dist(sub_dfm_new, method = "euclidean")
hc <- hclust(dist.mat, method = "ward.D2")
dend <- hc %>% 
  as.dendrogram() %>% 
  set("labels", l1_abbr, order_value = T) %>%
  set("labels_colors", l1_colors, order_value = T)
hc$labels <- l1_abbr
hc.dend <- fviz_dend(hc, cex = 0.7, lwd=0.5, show_labels=T,
                     label_cols = plyr::mapvalues(bawe_l1$L1[hc$order],
                                                  l1s, c("black", "orange", "red", "blue", "grey")),
                     type="rectangle")


# Hierarchical Clustering with Aggregated Data
bawe_l1_all <- bawe.meta %>%
  filter(L1 %in% l1s) %>% 
  arrange(id)
sub_dfm_l1_all <- sub_dfm[bawe_l1_all$id,]
sub_dfm_agg <- sub_dfm_l1_all %>%
  mutate(L1 = bawe_l1_all$L1) %>% 
  group_by(L1) %>% 
  summarize_all(mean) %>%
  column_to_rownames("L1")
dist.mat <- dist(sub_dfm_agg, method = "euclidean")
hc <- hclust(dist.mat, method = "ward.D2")
dend <- hc %>% as.dendrogram()
hc$labels <- c("Cantonese", "English", "French", "German", "Japanese")
hc.dend2 <- fviz_dend(hc, cex = 0.5, lwd=0.5, show_labels=T,
                      label_cols =  plyr::mapvalues(row.names(sub_dfm_agg)[hc$order],
                                                    l1s, c("black", "orange", "red", "blue", "grey")),
                      type="rectangle")


# K-means Clustering with PCA
set.seed(42)
l1s <- c("English", "Chinese Cantonese", "Japanese", "French", "German")
bawe_l1 <- bawe.meta %>%
  filter(L1 %in% l1s) %>% 
  group_by(L1) %>% 
  sample_n(40) %>% 
  arrange(id)
sub_dfm_l1 <- sub_dfm[bawe_l1$id,]
zero_var_cols <- which(apply(sub_dfm_l1, 2, var)==0)
sub_dfm_new <- sub_dfm_l1[,-zero_var_cols]
l1_abbr <- plyr::mapvalues(bawe_l1$L1, l1s, c("E", "C", "J", "F", "G"))
l1_colors <- plyr::mapvalues(bawe_l1$L1, l1s, c("black", "orange", "red", "blue", "grey"))
rownames(sub_dfm_new) <- paste(l1_abbr, 1:nrow(bawe_l1), sep="_")
km <- kmeans(sub_dfm_new, 4)
factoextra::fviz_cluster(km, data = sub_dfm_new)
km_pca <- prcomp(sub_dfm_new)
round(factoextra::get_eigenvalue(km_pca), 1) %>% head
coord_df <- data.frame(km_pca$x[,1:2]) %>%
  mutate(L1 = bawe_l1$L1) %>%
  mutate(Cluster = as.factor(paste0("Cluster ", km$cluster)))
kmeans.plot <- ggplot(coord_df) +
  geom_vline(xintercept = 0) +
  geom_hline(yintercept = 0) +
  geom_point(aes(x = PC1, y = PC2, fill = L1), size = 2, shape = 21, alpha = .75) +
  scale_fill_manual(values = c("orange", "black", "blue", "grey", "red")) +
  xlab(paste0("Dimension 1")) +
  ylab("Dimension 2") +
  theme_linedraw() +
  theme(panel.grid.major.x = element_blank()) +
  theme(panel.grid.minor.x = element_blank()) +
  theme(panel.grid.major.y = element_blank()) +
  theme(panel.grid.minor.y = element_blank()) +
  theme(legend.position="top") +
  facet_grid(~Cluster)
df.km <- data.frame(L1 = bawe_l1$L1, cluster = km$cluster)
df.km <- as.data.frame.matrix(table(df.km$cluster, df.km$L1))
df.km$Size <- km$size
df.km$tss <- km$withinss
rownames(df.km) <- paste("Cluster", 1:4)


# Train-Test Split
set.seed(42)
sub_dfm_new <- sub_dfm_l1[,-zero_var_cols] %>% mutate(L1 = bawe_l1$L1)
rownames(sub_dfm_new) <- bawe_l1$id
train_idx <- bawe_l1 %>%
  group_by(L1) %>% 
  sample_n(30)
train_dfm <- sub_dfm_new[train_idx$id,] %>% 
  filter(L1 != "English") %>% 
  select(L1, everything())
test_dfm <- sub_dfm_new[!(rownames(sub_dfm_new) %in% train_idx$id),] %>% 
  filter(L1 != "English") %>% 
  select(L1, everything())


# Multinomial Lasso Regression
cv_fit <- cv.glmnet(as.matrix(train_dfm[, -1]), train_dfm[, 1],
                    family = "multinomial",type.multinomial = "grouped")
plot(cv_fit)
lambda_min <- cv_fit$lambda.min
lambda_lse <- cv_fit$lambda.1se
lasso_pred <- predict(cv_fit, newx = as.matrix(test_dfm[,-1]), s = lambda_lse,
                      type="class")


# Random Forest
set.seed(1234)
rf.model <- randomForest(formula = as.factor(L1) ~ ., data = train_dfm, mtry=5)
print(rf.model)
rf.pred <- predict(rf.model, newdata=test_dfm[,-1], type="class")


# Evaluating Both Models
cm.lasso <- confusionMatrix(as.factor(as.character(lasso_pred)),
                            as.factor(test_dfm$L1), mode="everything")
cm.rf <- confusionMatrix(as.factor(rf.pred), as.factor(test_dfm$L1),
                         mode="everything")
f1.df <- cbind(as.data.frame.matrix(cm.lasso$byClass)$F1,
               as.data.frame.matrix(cm.rf$byClass)$F1)
average.f1 <- colMeans(f1.df)
accuracy <- c(cm.lasso$overall["Accuracy"], cm.rf$overall["Accuracy"])
metrics.df <- rbind(metrics.df,average.f1,accuracy)*100
rownames(metrics.df) <- c("Chinese Cantonese", "French", "German",
                          "Japanese", "Average F1", "Accuracy")
colnames(metrics.df) <- c("Multinomial Lasso", "Random Forest")