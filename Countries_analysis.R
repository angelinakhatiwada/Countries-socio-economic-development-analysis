#' Application of Statistical Learning techniques for countries socio-economic development analysis
#' Author: Angelina Khatiwada
#' Date: 16-November-2021

#============== PACKAGES REQUIRED =================================
library(skimr)
library(readr)
library(plyr)
library(dplyr)
library(purrr)
library(VIM)
library(ggplot2)
library(caret)
library(grid)
library(gridExtra)
library("epitools")
library(pROC)
library(MASS)
library(class)
library(gmodels)
library(randomForest)
library(missForest)
library(dplyr)
library(stringr)
library(FactoMineR)
library(factoextra)
library('neuralnet')
library(tidyverse)
library(corrplot)
library(factoextra)
library("reshape2")
library("dendextend")
library(solitude)
library(nnet)
library(class)

#============== DATA IMPORT AND PREPROCESSING =====================

un_data <- read_csv("https://raw.githubusercontent.com/angelinakhatiwada/Countries-socio-economic-development-analysis/main/country_indicators.csv")
un_data <- as.data.frame(un_data)

un_data[un_data == -99] <- NA
un_data[un_data == "..."] <- NA
un_data[un_data == ".../..."] <- NA
un_data[un_data == "~0"] <- 0
un_data[un_data == "-~0.0"] <- 0
un_data[un_data == "~0.0"] <- 0

un_data[, -c(1:2, 55)] <- sapply(un_data[, -c(1:2, 55)],as.numeric)

dim(un_data)
skim(un_data)

#Dealing with missing values

un_data$`Net Official Development Assist. received (% of GNI)` <- NULL
un_data$MissCount = (rowSums(is.na(un_data))/dim(un_data)[2])
dim(un_data[un_data$MissCount > 0.5, ])

#un_data[un_data$MissCount > 0.5, ]

un_data_filtered <- un_data[!un_data$MissCount > 0.5, ]

un_data_filtered$Region <- as.factor(un_data_filtered$Region)
un_data_filtered$Development <- as.factor(un_data_filtered$Development)
summary(un_data_filtered$Region)
summary(un_data_filtered$Development)
un_data_filtered$MissCount <- NULL

skim(un_data_filtered)

#============== MISSING VALUES IMPUTATION =========================

un <- missForest(un_data_filtered[, -c(1:2, 54)], verbose = TRUE)

summary(un$ximp)

un$OOBerror

un_final <- un$ximp
un_final$Country <- un_data_filtered$country
un_final$Region <- un_data_filtered$Region
un_final$Development <- un_data_filtered$Development


#============== BASIC DESCRIPTIVE ANALYSIS ========================
un_renamed <- un_final

names(un_renamed)[1:51] <- c("I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10",
                             "I11", "I12", "I13", "I14", "I15", "I16", "I17", "I18", "I19", "I20",
                             "I21", "I22", "I23", "I24", "I25", "I26", "I27", "I28", "I29", "I30",
                             "I31", "I32", "I33", "I34", "I35", "I36", "I37", "I38", "I39", "I40",
                             "I41", "I42", "I43", "I44", "I45", "I46", "I47", "I48", "I49", "I50", "I51")

rownames(un_renamed) <- un_renamed$Country
un_renamed$Country <- NULL

Mu <- colMeans(un_renamed[, -c(52, 53)])
sigma <- apply(un_renamed[, -c(52, 53)], 2, sd)
descriptive<-round(cbind(M, sigma),2)
descriptive

un_renamed[, -c(52, 53)] <- scale(un_renamed[, -c(52, 53)], )

# Checking if there is any feature(s) with near zero variance
nzv <- nearZeroVar(un_renamed[, -c(52,53)], saveMetrics = TRUE)
sum(nzv$nzv == TRUE)

#============== CORRELATION MATRIX ================================

M<-cor(un_renamed[, -c(52, 53)])
M

corrplot(M, method = 'circle', tl.col="black", order = "AOE")

#============== PCA ===============================================

#computing PCA
res.pca <- prcomp(un_renamed[, -c(52, 53)])

# New correlation matrix
pca_df <- data.frame(res.pca$x)
M_pca<-cor(pca_df)
corrplot(M_pca, method = 'circle', tl.col="black", order = "AOE")

# Eigenvalues
eig.val <- get_eigenvalue(res.pca)
eig.val

# Results for Variables
res.var <- get_pca_var(res.pca)
res.var$coord          # Coordinates
res.var$contrib        # Contributions to the PCs
res.var$cos2           # Quality of representation 

# Loadings
loadings <- res.pca$rotation
#write.csv(loadings,'loadings.csv')

# Results for countries
res.ind <- get_pca_ind(res.pca)
res.ind$coord          # Coordinates
res.ind$contrib        # Contributions to the PCs
res.ind$cos2           # Quality of representation 

fviz_eig(res.pca)

fviz_pca_ind(res.pca,
             col.ind = 'cos2', # Color by the quality of representation
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE,
             max.overlaps =100 # Avoid text overlapping
)


fviz_pca_var(res.pca,
             col.var = "contrib", # Color by contributions to the PC
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE     # Avoid text overlapping
)


fviz_pca_biplot(res.pca, repel = TRUE,
                col.var = "#2E9FDF", # Variables color
                col.ind = "#696969"  # Individuals color
)


fviz_pca_ind(res.pca,
             col.ind = un_renamed$Development,
             #addEllipses = TRUE, # Concentration ellipses
             palette = c("#00AFBB", "#E7B800", "#FC4E07", '#33CC33'),
            #ellipse.type = "confidence",
             legend.title = "Groups",
             repel = TRUE
)

#============== K-MEANS ===========================================

# K-Means
set.seed(123)

#optimal number of clusters
fviz_nbclust(un_renamed[, -c(52,53)], kmeans, nstart =5, method = "wss") + theme_minimal()+
  geom_vline(xintercept = 4, linetype = 2) 

fviz_nbclust(un_renamed[, -c(52,53)], kmeans, nstart = 5, method = "silhouette") + theme_minimal()

kmeans_fit <- kmeans(un_renamed[, -c(52,53)], 4, nstart = 5)

# plot the clusters
fviz_cluster(kmeans_fit, data = un_renamed[, -c(52,53)],
             palette = c("#00AFBB", "#E7B800", "#FC4E07", '#33CC33'),
             geom = c("point"),ellipse.type = "euclid",
             ggtheme = theme_minimal())

kmeans_table <- data.frame(kmeans_fit$size, kmeans_fit$centers)
kmeans_df <- data.frame(Cluster = kmeans_fit$cluster, un_final)
head(kmeans_df)
#write.csv(kmeans_df,'kmeans_df.csv')

#plotting association with region and development level

kmeans_df$Cluster <- as.factor(kmeans_df$Cluster)
p1 <- ggplot(kmeans_df, aes(x=Region, fill=Cluster)) +
  geom_bar(stat="count",  alpha = 0.7) + theme_minimal() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  scale_fill_manual(values=c("#00AFBB", "#E7B800", "#FC4E07", '#33CC33'))

p2 <- ggplot(kmeans_df, aes(x=Region, fill=Cluster)) +
  geom_bar(position="fill", alpha = 0.6) + theme_minimal() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  scale_fill_manual(values=c("#00AFBB", "#E7B800", "#FC4E07", '#33CC33'))

p3 <- ggplot(kmeans_df, aes(x=Development, fill=Cluster)) +
  geom_bar(position="fill", alpha = 0.6) + theme_minimal() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  scale_fill_manual(values=c("#00AFBB", "#E7B800", "#FC4E07", '#33CC33'))

p4 <- ggplot(kmeans_df, aes(x=Cluster, fill=Development)) +
  geom_bar(position="fill",  alpha = 0.7) + theme_minimal() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  scale_fill_manual(values=c("#00AFBB", "#E7B800", "#FC4E07", '#33CC33'))

p1
p2
p3
p4

#============== HIERARCHICAL CLUSTERING ============================


d <- dist(un_renamed[, -c(52, 53)], method = "euclidean") # Euclidean distance matrix
H <- hclust(d, method="ward.D2") #Ward linkage
plot(H)

dendro <- as.dendrogram(H)
dendro.col <- dendro %>%
  set("branches_k_color", k = 4) %>%
  set("branches_lwd", 0.6) %>%
  set("labels_colors", 
      value = c("darkslategray")) %>% 
  set("labels_cex", 0.5)
ggd1 <- as.ggdend(dendro.col)
ggplot(ggd1, theme = theme_minimal()) +
  labs(x = "Num. observations", y = "Height", title = "Dendrogram, k = 4") +
geom_hline(yintercept = 32, linetype = 2) 

#============== ADDITIONAL PLOTS ===================================

p5 <- ggplot(data = un_final, aes(un_final$`Surface area (km2)`, color = un_final$Development))+
geom_freqpoly(binwidth = 5, size = 1)
p5

p6 <- ggplot(data = un_final, aes(un_final$`Economy: Agriculture (% of GVA)`, color = un_final$Development))+
geom_freqpoly(binwidth = 5, size = 1)
p6

p7 <- ggplot(data = un_final, aes(un_final$`Unemployment (% of labour force)`, color = un_final$Development))+
  geom_freqpoly(binwidth = 5, size = 1)
p7

p8 <- ggplot(data = un_final, aes(un_final$`Life expectancy at birth (females, years)`, color = un_final$Development))+
  geom_freqpoly(binwidth = 5, size = 1)
p8

#============== OUTLIERS DETECTION =================================

un_renamed_class <- un_renamed

# Boxplots and IQR

draw_boxplot <- function(){ 
  un_renamed_class %>%  
    pivot_longer(1:51, names_to="indicators") %>%  
    ggplot(aes(indicators, value, fill=indicators)) + 
    geom_boxplot() 
}

draw_boxplot()

#Ouliers with an Isolation Forest

iforest <- isolationForest$new(sample_size = 200, num_trees = 500,replace = TRUE)
iforest
iforest$fit(un_renamed_class[, -c(52,53)])

scores_train = un_renamed_class[, -c(52,53)] %>%
  iforest$predict() #%>%
#arrange(desc(anomaly_score))

un_renamed_class$iforest_anomaly_score <- scores_train$anomaly_score

un_renamed_class[order(-un_renamed_class$iforest_anomaly_score),]
skim(un_renamed_class$iforest_anomaly_score)
quantile(un_renamed_class$iforest_anomaly_score, probs = c(0.85))

un_renamed_class$iforest_outlier <- as.factor(ifelse(un_renamed_class$iforest_anomaly_score >=0.60, "outlier", "normal"))

#Ouliers with PCA inverse

res.pca <- prcomp(un_renamed_class[, -c(52:55)])

#reconstruction loss
nComp = 12
Xhat = res.pca$x[,1:nComp] %*% t(res.pca$rotation[,1:nComp])

original = as.matrix(un_renamed_class[, -c(52:55)])

my_range <- 1:214

#calculating MSE between original and reconstructed values

losses = c()
for(i in my_range) {                                        
  #print(i)
  mse_loss = mean((original[i, ] - Xhat[i, ])^2)
  #print(mse_loss)
  losses <- c(losses, mse_loss)
}

un_renamed_class$pca_losses <- losses
un_renamed_class[order(-un_renamed_class$pca_losses), ]

skim(un_renamed_class$pca_losses)
quantile(un_renamed_class$pca_losses, probs = c(0.85))

un_renamed_class$pca_outlier <- as.factor(ifelse(un_renamed_class$pca_losses >=0.30, "outlier", "normal"))

#combining the result of Isolation Forest and PCA

un_renamed_class[un_renamed_class$pca_outlier == 'outlier', ][, c(52:57)]
un_renamed_class[un_renamed_class$iforest_outlier == 'outlier', ][, c(52:57)]

outliers  <- un_renamed_class[(un_renamed_class$iforest_outlier == 'outlier') &
                                (un_renamed_class$pca_outlier == 'outlier'), ]

rownames(outliers)
outliers[, c(52:57)]

normal <- un_renamed_class[(un_renamed_class$iforest_outlier == 'normal') |
                             (un_renamed_class$pca_outlier == 'normal'), ]

dim(un_renamed_class)
dim(outliers)
dim(normal)

dim(outliers)/dim(normal)
#write.csv(outliers,'outliers.csv')

#============== TRAIN/TEST SPLIT FOR CLASSIFICATION ================

#train-test split
set.seed(12345)
split_train_test <- createDataPartition(un_renamed_class$Development,p=0.65,list=FALSE)
train.data<- un_renamed_class[split_train_test,]
test.data<-  un_renamed_class[-split_train_test,]

dim(train.data)
dim(test.data)

#============== RANDOM FOREST ======================================

#Dataset will all the indicators and the observations - train/test split

set.seed(1234)
# training
rf1 <- randomForest(Development ~., data = train.data[, -c(52)], importance=TRUE)
rf1

plot(rf1)
grid (NULL,NULL, lty = 6, col = "cornsilk2") 

varImpPlot(rf1)

predictions <- predict(rf1, test.data[, -c(52)], type = "class")
# Accuracy and other metrics
confusionMatrix(predictions, test.data$Development)

#Dataset will all the indicators and the observations - entire set

rf2 <- randomForest(Development ~., data = un_renamed_class[, -c(52)], importance=TRUE)
rf2
plot(rf2)
grid (NULL,NULL, lty = 6, col = "cornsilk2") 
varImpPlot(rf2)

#Dataset will all the observations and 12 PCs - entire set
head(pca_df)
pca_df$Development <- un_renamed_class$Development


rf3 <- randomForest(Development ~., data = pca_df[, -c(13:51)], importance=TRUE)
rf3
plot(rf3) 
grid (NULL,NULL, lty = 6, col = "cornsilk2") 
varImpPlot(rf3)

#Dataset will all the indicators and outliers removed - 200 obs

rf4 <- randomForest(Development ~., data = normal[, -c(52, 54:57)], importance=TRUE)
rf4
plot(rf4)
grid (NULL,NULL, lty = 6, col = "cornsilk2") 
varImpPlot(rf4)

#============== OTHER MODELS ======================================

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}

#K-NN

#train set prediction
prc_train_pred <- knn(train = train.data[, -c(52:53)], test = train.data[, -c(52:53)], cl = train.data$Development, k=8)
prc_train_pred
tab <- table(prc_train_pred,train.data$Development)
accuracy(tab)

#test set prediction
prc_test_pred <- knn(train = train.data[, -c(52:53)], test = test.data[, -c(52:53)], cl = train.data$Development, k=8)
prc_test_pred
tab <- table(prc_test_pred,test.data$Development)

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tab)

# ANN

softplus <- function(x) log(1 + exp(x))
nn=neuralnet(Development~.,  
             data=train.data[, -c(52)], hidden=c(10,4), linear.output = FALSE,
             act.fct = softplus)

prediction <- data.frame(neuralnet::compute(nn, data.frame(train.data[, -c(52:53)]))$net.result)
labels <- c("developed", "developing", "economies in transition", "least developed") 
prediction_label <- data.frame(max.col(prediction)) %>%  
  mutate(prediction=labels[max.col.prediction.]) 
tab_train <- table(train.data$Development,  prediction_label$prediction) 
accuracy(tab_train)

prediction <- data.frame(neuralnet::compute(nn, data.frame(test.data[, -c(52:53)]))$net.result)
labels <- c("developed", "developing", "economies in transition", "least developed") 
prediction_label <- data.frame(max.col(prediction)) %>%  
  mutate(prediction=labels[max.col.prediction.]) 
tab_test <- table(test.data$Development,  prediction_label$prediction) 
accuracy(tab_test)


#Logistic Regression

full_log_model <-  nnet::multinom(Development ~., data = train.data[, -c(52)])
summary(full_log_model)

predicted.classes <- full_log_model %>% predict(test.data[, -c(52,53)])
head(predicted.classes)

mean(predicted.classes == test.data$Development)

table(Predicted = predicted.classes, Actual = test.data$Development)

confusionMatrix(
  as.factor(predicted.classes),
  as.factor(test.data$Development),
  positive = "1" 
)



