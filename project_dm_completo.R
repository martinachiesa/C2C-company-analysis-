# rpart

dati <- read.csv('dati.csv', sep = ',', dec = '.', stringsAsFactors = T, na.strings = c('NA', 'NaN', -1))

#creo il target
dati$venditore=ifelse(dati$productsSold > 0,"c1","c0")
table(dati$venditore)
table(dati$venditore) / nrow(dati)#il 98% di non venditori e solo il 2% di venditori
#analizziamo brevemente le variabili del dataset
str(dati)
library(funModeling)
library(dplyr)
library(survival)
status(dati)#  La variabile type assume una sola modalità == 0 variance, inoltre tolgo identifierhash
dati$identifierHash <- NULL
dati$type <- NULL
dati$productsSold <- NULL
#modifico la variabile civilityGenderID
dati$civilityGenderId <- as.factor(dati$civilityGenderId)


#PREPROCESSING: NA, COLL, NZV
#VALORI MANCANTI
sapply(dati, function(x)(sum(is.na(x)))) #conteggio dei valori mancanti

numeric <- sapply(dati, function(x) is.numeric(x))
numeric <-dati[, numeric]
head(numeric)

library(mice)

imp <- mice(numeric, m=5, maxit=10, meth='pmm', seed=500)

completedData <- complete(imp,1)  
venditore=dati$venditore

isfactor <- sapply(dati, function(x) is.factor(x))
factor <- dati[,isfactor]

dati_imput=cbind(completedData, factor, venditore)
names(dati_imput)
sapply(dati_imput, function(x)(sum(is.na(x)))) 

#save(dati_imput, file = "C:/Users/chies/OneDrive/Desktop/DATA MINING/progetto/dati_imput.Rdata")
#load("dati_imput.RData")
str(dati_imput)

#collinearità
library(caret)
numeric <- sapply(dati_imput, function(x) is.numeric(x))
numeric <-dati_imput[, numeric]

R=cor(numeric)
R
correlatedPredictors = findCorrelation(R, cutoff = 0.95, names = TRUE)
correlatedPredictors 

require(corrgram)
corrgram(numeric,lower.panel = panel.cor, cex=1, cex.labels = 1)#asYears asmonth
dati_imput[,correlatedPredictors] = NULL
head(dati_imput)

# correlazione variabili qualitative
dati_imput$gender <- dati_imput$civilityGenderId <- NULL
status(dati_imput)

#near 0 var
nzv = nearZeroVar(dati_imput, saveMetrics = TRUE)
nzv
table(dati_imput$productsPassRate)
dati_imput$productsPassRate <- NULL
dati_imput$productsListed <- NULL

dati_imput$country <- NULL
#dobbiamo sistemare CountryCode
dd <- sort(table(dati_imput$countryCode), decreasing=T); dd

dati_imput$optimal_country=dati_imput$countryCode
dati_imput$optimal_country = as.character(dati_imput$optimal_country)
`%notin%` <- Negate(`%in%`)
dati_imput$optimal_country[ which(dati_imput$optimal_country %notin% c('fr','it','us','gb'))] <- 'other'
dati_imput$optimal_country = as.factor(dati_imput$optimal_country)
dati_imput$optimal_country <- droplevels(dati_imput$optimal_country)
dati_imput$countryCode <- NULL

table(dati_imput$optimal_country)

#under sampling by groups
library(ROSE)
data_balanced <- ovun.sample(venditore~., data=dati_imput,
                                p=0.3,
                               seed=1, method="under")$data

table(data_balanced$venditore)

#dividiamo il dataset
library(caret)
set.seed(1)
cpart=createDataPartition(y=data_balanced$venditore,times=1,p=.6)
train.df=data_balanced[cpart$Resample1,]
test.df=data_balanced[-cpart$Resample1,]
cpart2=createDataPartition(y=test.df$venditore,times=1,p=.125)
score.df=test.df[cpart2$Resample1,]
test.df=test.df[-cpart2$Resample1,]

#save(train.df, file = "C:/Users/chies/OneDrive/Desktop/DATA MINING/progetto/train.df.Rdata")
#save(test.df, file = "C:/Users/chies/OneDrive/Desktop/DATA MINING/progetto/test.df.Rdata")
#save(score.df, file = "C:/Users/chies/OneDrive/Desktop/DATA MINING/progetto/score.df.Rdata")

str(train.df)
str(test.df)

###
#1. ALBERI
###

#alberi come modello e come model selector x scegliere le variabili più importanti
set.seed(1)
cvCtrl <- trainControl(method = "cv", number=10, search="grid", classProbs = TRUE, summaryFunction=twoClassSummary)
tree <- train(venditore~. ,data=train.df, 
              method = "rpart",
              tuneLength = 10,
              trControl = cvCtrl,
              metric="Spec")
# best sens using best cp
tree

# final model
plot(tree)
varImp(object=tree)
plot(varImp(object=tree), main="train tuned - Variable Importance")

yhat.tree = predict(tree, newdata = train.df)
table(yhat.tree, train.df$venditore)/nrow(train.df)

# uso random forest come selector, più robusto
tunegrid_rf <- expand.grid(.mtry=c(2:7)) 
#griglia intorno al valori di radice di p: p^(0.5)=16^(0.5)=4

control <- trainControl(method="cv", number=10, search="grid", summaryFunction = twoClassSummary, classProbs = TRUE)
rf <-  train(venditore~., data=train.df, method="rf", tuneGrid=tunegrid_rf, ntree=250, trControl=control, metric="Spec")
confusionMatrix(rf)

print(rf)
varImp(object = rf)
plot(varImp(object=rf), main="train tuned - Variable Importance")

# select most important var (importance > 5)
list=c("socialNbFollowers", "daysSinceLastLogin","socialProductsLiked","hasProfilePicture",
       "seniority","socialNbFollows","productsWished","productsBought","hasAnyApp","venditore")

train_modsel=train.df[,list]
test_modsel=test.df[,list]

#save(train_modsel, file = "C:/Users/chies/OneDrive/Desktop/DATA MINING/progetto/train_modsel.Rdata")
#save(test_modsel, file = "C:/Users/chies/OneDrive/Desktop/DATA MINING/progetto/train_modsel.Rdata")


###
#2. BAGGING (no preprocessing, sono alberi)
###
head(train.df)
seed <- 1
set.seed(seed)
control <- trainControl(method="cv", number=10, search="grid", summaryFunction = twoClassSummary, 
                        classProbs = TRUE)
tunegrid_bg <- expand.grid(.mtry=c(15))#1:5
bagging <- train(venditore~., data=train.df, method="rf", tuneGrid=tunegrid_bg, ntree=250, 
                 trControl=control, metric="Spec")
#un rf con 22 variabili è un bagging

yhat.bag = predict(bagging, newdata = train.df)
table(yhat.bag, train.df$venditore)/nrow(train.df)


###
#3. GLM 
###

# mod base GLM con dataset mod sel (coll, NA, 0 variance, model selection)
set.seed(1)
ctrl =trainControl(method="cv", number = 10, classProbs = T,
                   summaryFunction=twoClassSummary)
glm=train(venditore~.,
          data=train_modsel,method = "glm",
          trControl = ctrl, tuneLength=5, trace=TRUE, na.action = na.pass, metric="Spec")

glm
confusionMatrix(glm)

#3b. modello GLM con dataset non preprocessato
set.seed(1)
ctrl =trainControl(method="cv", number = 10, classProbs = T,
                   summaryFunction=twoClassSummary)
glm_cor=train(venditore~.,
          data=train.df,method = "glm",
          trControl = ctrl, tuneLength=5, trace=TRUE, na.action = na.pass, 
          preProcess=c("corr"), metric="Spec")

glm_cor
confusionMatrix(glm_cor)
# metriche leggermenti migliori del precedente


###
#4. LASSO 
###


# (NA, collinearità, 0 variance)
set.seed(1)
ctrl =trainControl(method="cv", number = 10, classProbs = T,
                   summaryFunction=twoClassSummary)
grid = expand.grid(.alpha=1,.lambda=seq(0, 1, by = 0.1))
lasso=train(venditore~.,
            data=train.df, method = "glmnet",
            trControl = ctrl, tuneLength=5, na.action = na.pass,
            tuneGrid=grid, metric="Spec")
lasso
plot(lasso)
confusionMatrix(lasso)



###
#5. NAIVE BAYES 
###


#(NA, collinearità, 0 problem, near 0 variance)

qplot(optimal_country, color=venditore, data=train.df, geom='density')
qplot(socialNbFollowers, color=venditore, data=train.df, geom='density')
qplot(hasAnyApp, color=venditore, data=train.df, geom='density')
qplot(daysSinceLastLogin, color=venditore, data=train.df, geom='density')

set.seed(1)
ctrl =trainControl(method="cv", number = 10, classProbs = T,
                   summaryFunction=twoClassSummary)
naivebayes=train(venditore~.,
                 data=train.df,method = "naive_bayes",
                 trControl = ctrl, tuneLength=5, na.action = na.pass, metric="Spec") 
naivebayes
confusionMatrix(naivebayes)


#5b. Naive Bayes con model sel
# non è necessaria la model sel poichè non affetto da input irrilevanti, ma comunque consigliabile

set.seed(1)
ctrl =trainControl(method="cv", number = 10, classProbs = T,
                   summaryFunction=twoClassSummary)
naivebayes_sel=train(venditore~.,
                 data=train_modsel,method = "naive_bayes",
                 trControl = ctrl, tuneLength=5, na.action = na.pass, metric="Spec") 
naivebayes_sel
confusionMatrix(naivebayes_sel)




###
#6. GRADIENT BOOSTING
###


#(no preprocessing)
set.seed(1)
ctrl =trainControl(method="cv", number = 10, classProbs = T,
                   summaryFunction=twoClassSummary)
mtryGrid <- expand.grid(interaction.depth = seq(1, 7, by = 2),
                        n.trees = seq(100, 1000, by = 50),
                        n.minobsinnode = 10,
                        shrinkage = c(0.01, 0.1))

expand.grid(n.trees=c(10,20,60),shrinkage=c(0.05,0.1,0.5),n.minobsinnode = c(3,5),interaction.depth=c(3,5))
boost1=train(venditore~.,
             data=train.df,method = "gbm", 
             trControl = ctrl, tuneLength=5, 
             tuneGrid=mtryGrid,
             metric="Spec")
boost1
plot(boost1)
confusionMatrix(boost1)



###
#7. PLS
###


library(pls)
set.seed(1)
Control=trainControl(method= "cv",number=10, classProbs=TRUE,
                     summaryFunction=twoClassSummary)
pls=train(venditore~. , data=train.df , method = "pls", 
          trControl = Control, tuneLength=5, metric="Spec")
pls
plot(pls)
confusionMatrix(pls)




###
#8. KNN 
###

#lavora con var quantitative

# KNN crossvalidato
set.seed(1)
ctrl = trainControl(method="cv", number=10, classProbs=T,
                    summaryFunction=twoClassSummary)
# number: Either the number of folds or number of resampling iterations

grid = expand.grid(k=seq(5,20,3))
knn_scale=train(venditore~.,
          data=train_modsel,method = "knn",
          trControl = ctrl, tuneLength=5, na.action = na.pass, metric="Spec",
          tuneGrid=grid, preProcess=c("scale"))
knn_scale
plot(knn_scale)
confusionMatrix(knn_scale)

#8b. KNN con Bootstrap
set.seed(1)
knn_boot <- train(venditore ~., data=train_modsel,
                 method = "knn",
                 preProcess = c("center", "scale"),
                 tuneLength = 5, 
                 trControl = trainControl(method = "boot",
                                          summaryFunction = twoClassSummary, 
                                          classProbs = TRUE), 
                                          metric="Spec")
knn_boot
plot(knn_boot)
confusionMatrix(knn_boot)

#8c. KNN con principal components
set.seed(1)
cvCtrl <- trainControl(method = "cv", number=10, searc="grid", 
                       summaryFunction = twoClassSummary, 
                       classProbs = TRUE)

knn_pc <- train(venditore ~., data=train_modsel,
                 method = "knn", tuneLength = 5,
                 preProcess = c("pca"),
                 metric="Spec",
                 trControl = cvCtrl)
knn_pc
plot(knn_pc)
confusionMatrix(knn_pc)





###
#9 NEURAL NETS
###


# come preprocessing devo scalare var

library(nnet)
set.seed(1)
y=ifelse(train_modsel$venditore=="c1",1,0)
train_net = train_modsel
head(train_net)
train_net$hasProfilePicture=ifelse(train_net$hasProfilePicture=="True",1,0)
train_net$hasAnyApp=ifelse(train_net$hasAnyApp=="True",1,0)

head(train_net)
#MLP
mynet <- nnet(train_net[,-10], y,entropy=T, hidden=1, size=3,decay=0.1, maxit=2000, trace=T, metric="Spec")
mynet

library(NeuralNetTools)
plotnet(mynet, alpha=0.6)

mynet.pred <- as.numeric(predict(mynet, train_net[,-10], type='class'))
table(mynet.pred,y)/nrow(train_net)


#net whith default parameters grid
library(caret)
set.seed(1)
metric <- "Spec"
ctrl = trainControl(method="cv", number=5, classProbs = TRUE, search = "grid", summaryFunction = twoClassSummary)

nnet_grid <- train(train_net[-10], as.factor(train_net$venditore),
                     method = "nnet",
                     preProcess = c("range","corr"), 
                     metric=metric, trControl=ctrl, 
                     trace = TRUE, # use true to see convergence
                     maxit = 300)

print(nnet_grid)
plot(nnet_grid)
confusionMatrix(nnet_grid)

library(caret)
getTrainPerf(nnet_grid) 
confusionMatrix(nnet_grid)


# net with grid search near the default ESTIMATED tuned parameters: size=1, decay=0.1

set.seed(1)
metric <- "Spec"
ctrl = trainControl(method="cv", number=10, search="grid", summaryFunction = twoClassSummary)
tunegrid <- expand.grid(size=c(1:3), decay = c(0.001, 0.01, 0.05 , .1, 0.2, .3))
nnet_near <- train(train_net[-10], train_net$venditore,
                         method = "nnet",
                         preProcess = c("scale"), 
                         tuneLength = 10, metric=metric, trControl=ctrl, tuneGrid=tunegrid,
                         trace = TRUE,
                         maxit = 300)
print(nnet_near)
plot(nnet_near)
getTrainPerf(nnet_near)
confusionMatrix(nnet_near)

# best result: size=1, decay=0.3


save(tree, rf, glm, glm_cor, bagging, lasso, boost1, knn_scale, knn_boot, knn_pc, nnet_grid, pls, naivebayes,  naivebayes_sel, file = "modelli.RData")
#load('modelli.Rdata')
#load('train.df')
#load('test.df')
#load('train_modsel')
#load('test_modsel')

# modelli competitivi:
# perceptron, rete mlv

# confronto con e senza scale
# tree come model selector
# griglia di parametri intorno default
# griglia intorno parametri stimati
# pca come preprocessing

# nnet (but also Knn) require scaling X, check collin and model selection!!
# tree as model selector for the net

# size= number of units in the hidden layer
# Weight decay= lambda, penalizes C(wij), the sum of squares of the weights wij
# scale inputs

# dont using random searh on black box



#STEP 2

#curve ROC
test2 = test.df
test2$y=ifelse(test.df$venditore=="c1",1,0)
test2$hasProfilePicture=ifelse(test2$hasProfilePicture=="True",1,0)
test2$hasAnyApp=ifelse(test2$hasAnyApp=="True",1,0)
test.df$tree=predict(tree,test.df, type="prob")[,2]
nnet_grid_old_pr_1=predict(nnet_grid,test2, "prob")[,2]

tree_old_pr_1=predict(tree,test.df, type="prob")[,2]
rf_old_pr_1=predict(rf,test.df, "prob")[,2]
glm_cor_old_pr_1=predict(glm_cor,test.df, "prob")[,2]
bagging_old_pr_1=predict(bagging,test.df, "prob")[,2]
naivebayes_sel_old_pr_1=predict(naivebayes_sel,test.df, "prob")[,2]
lasso_old_pr_1=predict(lasso,test.df, "prob")[,2]
boost1_old_pr_1=predict(boost1,test.df, "prob")[,2]
knn_pc_old_pr_1=predict(knn_pc,test.df, "prob")[,2]
nnet_grid_old_pr_1=predict(nnet_grid,test2, "prob")[,2]
nnet_near_old_pr_1=predict(nnet_near,test2, "prob")[,2]


rho0=0.7
rho1=0.3
#1-2036/98913
true0 =0.9794163
true1 =0.02058375

den=tree_old_pr_1*(true1/rho1)+(1-tree_old_pr_1)*(true0/rho0)




########

library(pROC)
pred1_true= tree_old_pr_1*(true1/rho1)/den
pred0_true= (1-tree_old_pr_1)*(true0/rho0)/den
roc.tree=roc(venditore ~ pred1_true, data = test.df)
roc.tree

library(pROC)
roc.tree=roc(venditore ~ tree_old_pr_1, data = test.df)
roc.rf=roc(venditore ~ rf_old_pr_1, data = test.df)
roc.glm_corr=roc(venditore ~ glm_cor_old_pr_1, data = test.df)
roc.bagging=roc(venditore ~ bagging_old_pr_1, data = test.df)
roc.naivebayes_sel=roc(venditore ~ naivebayes_sel_old_pr_1, data = test.df)
roc.lasso=roc(venditore~ lasso_old_pr_1, data = test.df)
roc.boost1=roc(venditore~ boost1_old_pr_1, data = test.df)
roc.knn_pc=roc(venditore~ knn_pc_old_pr_1, data = test.df)
roc.nnet_grid=roc(venditore~ nnet_grid_old_pr_1, data = test2)
roc.nnet_near=roc(venditore~ nnet_near_old_pr_1, data = test2)

roc.tree
roc.rf
roc.glm_corr
roc.bagging
roc.naivebayes_sel
roc.lasso
roc.boost1
roc.knn_pc
roc.nnet_grid
roc.nnet_near

plot(roc.tree)
plot(roc.rf,add=T,col="red")
plot(roc.glm_corr,add=T,col="blue")
plot(roc.bagging,add=T,col="yellow")
plot(roc.naivebayes_sel,add=T,col="green")
plot(roc.lasso,add=T,col="darkmagenta")
plot(roc.boost1,add=T,col="pink")
plot(roc.knn_pc,add=T,col="purple")
plot(roc.nnet_grid, add=T, col='blue')
plot(roc.nnet_near, add=T, col='grey')


test.df$rf_old_pr_1 <- rf_old_pr_1
test.df$boost1_old_pr_1 <- boost1_old_pr_1
test.df$bagging_old_pr_1 <- bagging_old_pr_1



# CURVE LIFT
library(funModeling)
gain_lift(data = test.df, score = 'rf_old_pr_1', target = "venditore")
gain_lift(data = test.df, score = 'boost1_old_pr_1', target = "venditore")
gain_lift(data = test.df, score = 'bagging_old_pr_1', target = "venditore")

#ranom forest modello vincente

#STEP3
df=test.df[,-c(16,17,18,19)]
head(df)

den_rf=rf_old_pr_1*(true1/rho1)+(1-rf_old_pr_1)*(true0/rho0)
pred1_true_rf= rf_old_pr_1*(true1/rho1)/den_rf
pred0_true_rf=(1-pred1_true_rf)

df$venditore=ifelse(df$venditore=="c1","V","N")

df$rf <- pred1_true_rf 
head(df)

library(dplyr)
# for each threshold, find tp, tn, fp, fn and the sens=prop_true_M, spec=prop_true_R, precision=tp/(tp+fp)

thresholds <- seq(from = 0, to = 1, by = 0.01)
prop_table <- data.frame(threshold = thresholds, prop_true_V = NA,  prop_true_N = NA, true_V = NA,  true_N = NA , fn_V=NA)

for (threshold in thresholds) {
  pred <- ifelse(df$rf > threshold, "V", "N")  # be careful here!!!
  pred_t <- ifelse(pred == df$venditore, TRUE, FALSE)
  
  group <- data.frame(df, "pred" = pred_t) %>%
    group_by(venditore, pred) %>%
    dplyr::summarise(n = n())
  
  group_V <- filter(group, venditore == "V")
  
  true_V=sum(filter(group_V, pred == TRUE)$n)
  prop_V <- sum(filter(group_V, pred == TRUE)$n) / sum(group_V$n)
  
  prop_table[prop_table$threshold == threshold, "prop_true_V"] <- prop_V
  prop_table[prop_table$threshold == threshold, "true_V"] <- true_V
  
  fn_V=sum(filter(group_V, pred == FALSE)$n)
  # true M predicted as R
  prop_table[prop_table$threshold == threshold, "fn_V"] <- fn_V
  
  
  group_N <- filter(group, venditore == "N")
  
  true_N=sum(filter(group_N, pred == TRUE)$n)
  prop_N <- sum(filter(group_N, pred == TRUE)$n) / sum(group_N$n)
  
  prop_table[prop_table$threshold == threshold, "prop_true_N"] <- prop_N
  prop_table[prop_table$threshold == threshold, "true_N"] <- true_N
  
}

head(prop_table, n=10)


prop_table$n=nrow(df)

# false positive (fp_M) by difference of   n and            tn,                 tp,         fn, 
prop_table$fp_V=nrow(df)-prop_table$true_N-prop_table$true_V-prop_table$fn_V

# find accuracy
prop_table$acc=(prop_table$true_N+prop_table$true_V)/nrow(df)

# find precision
prop_table$prec_V=prop_table$true_V/(prop_table$true_V+prop_table$fp_V)

# find F1 =2*(prec*sens)/(prec+sens)
# prop_true_M = sensitivity

prop_table$F1=2*(prop_table$prop_true_V*prop_table$prec_V)/(prop_table$prop_true_V+prop_table$prec_V)

# verify not having NA metrics at start or end of data 
tail(prop_table)
head(prop_table)
# we have typically some NA in the precision and F1 at the boundary..put,impute 1,0 respectively 

library(Hmisc)
#impute NA as 0, this occurs typically for precision
prop_table$prec_V=impute(prop_table$prec_V, 1)
prop_table$F1=impute(prop_table$F1, 0)
tail(prop_table)
colnames(prop_table)
# drop counts, PLOT only metrics
prop_table2 = prop_table[,-c(4:8)] 
head(prop_table2)

# plot measures vs soglia
# before we must impile data vertically: one block for each measure
library(dplyr)
library(tidyr)
gathered=prop_table2 %>%
  gather(x, y, prop_true_V:F1)
head(gathered)


# grafico con tutte le misure 
library(ggplot2)
gathered %>%
  ggplot(aes(x = threshold, y = y, color = x)) +
  geom_point() +
  geom_line() +
  scale_color_brewer(palette = "Set1") +
  labs(y = "measures",
       color = "M: event\nR: nonevent")


## STEP 4
# follow sensitivity= prop true M...beccre i veri ricchi (soglie basse) 
# anche F1 ciferma soglie attorno a  0.20
table(score.df$venditore)

head(score.df)
rf_score <- predict(rf,score.df, "prob")[,2]

den_rf=rf_score*(true1/rho1)+(1-rf_score)*(true0/rho0)
score_true_rf= rf_score*(true1/rho1)/den_rf
pred0_true_rf=(1-score_true_rf)
head(score_true_rf)
score.df$rf <- score_true_rf

pred <- ifelse(score.df$rf > 0.05, "c1", "c0")
table(actual=score.df$venditore, pred)

library(caret)
confusionMatrix(as.factor(pred), as.factor(score.df$venditore), positive="c1")





