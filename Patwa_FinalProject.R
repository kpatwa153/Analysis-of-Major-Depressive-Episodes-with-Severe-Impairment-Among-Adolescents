#220:422 & 219:531 ADM&ML
#Final Project, Spring 2024
#Goal: Build Two Classifiers to identify "mdeSI" among adolescents
#Your Full Name: Kenil Patwa
library(caret)
library(corrplot)
library(ggplot2)
library(treemap)
library(tidyr)
library(MASS)
library(mixlm)
library(randomForest)
library(RColorBrewer)
library(plotly)
library(pROC)
da = read.csv("depression.csv", header = T)

dim(da) #6,000 by 12
str(da)

da$mdeSI = factor(da$mdeSI)
da$income = factor(da$income)
da$gender = factor(da$gender, levels = c("Male", "Female")) #Male is the reference group
da$age = factor(da$age)
da$race = factor(da$race, levels = c("White", "Hispanic", "Black", "Asian/NHPIs", "Other")) #white is the reference
da$insurance = factor(da$insurance, levels = c("Yes", "No")) #"Yes" is the reference group
da$siblingU18 = factor(da$siblingU18, levels = c("Yes", "No"))
da$fatherInHH = factor(da$fatherInHH)
da$motherInHH = factor(da$motherInHH)
da$parentInv = factor(da$parentInv)
da$schoolExp = factor(da$schoolExp, levels = c("good school experiences", "bad school experiences"))

(n = dim(da)[1])
set.seed(2024)
index = sample(1:n, 4500) #75% of training and 25% of test data
train = da[index,] 
test = da[-index,]
dim(train)
dim(test)

#Data Exploration and Features Selection

#here I am viewing the class, total unique values and length of unique values for each variable.

(sapply(da, class))
unique(da$gender); length(unique(da$gender))
unique(da$age); length(unique(da$age))
unique(da$race); length(unique(da$race))
unique(da$insurance); length(unique(da$insurance))
unique(da$income); length(unique(da$income))
unique(da$fatherInHH); length(unique(da$fatherInHH))
unique(da$motherInHH); length(unique(da$motherInHH))
unique(da$siblingU18); length(unique(da$siblingU18))
unique(da$parentInv); length(unique(da$parentInv))
unique(da$schoolExp); length(unique(da$schoolExp))

####Count of Depression cases per year####
table(da$mdeSI,da$year)
ggplot(data = da, aes(x = year, fill = mdeSI)) +
  geom_bar(position = "dodge") +
  labs(x = "Year", y = "Count", fill = "mdeSI") +
  ggtitle("Year-wise Count of Depression") +
  theme_minimal()

#Here we can see that the depression cases increased drastically from 2015 to 2016. But from 2016 to 2017, the cases of depression
#Stayed the same, but we saw that almost 3x people are not suffering from depression than those who are.
#One more observation is that for the year 2015, 2016 there is not a single case where mdeSI = No => everyone is suffering from depression.
#So next analysis is to find the change from 2015,2016 to 2017 which showed the trend that more people are not suffering from depression.

####Year Wise Analysis####
da3 = subset(da, !(year == 2017 & mdeSI == "Yes"))
da_2015 = subset(da, year == 2015)
da_2016 = subset(da, year == 2016)
da_2017 = subset(da, year == 2017)
da_2017_no = subset(da_2017, mdeSI == "No")

#Pie chart for variables: fatherInHH, motherInHH, siblingU18
value_2017_father = table(da_2017_no$fatherInHH)
value_2017_mother = table(da_2017_no$motherInHH)
value_2017_siblingU18 = table(da_2017_no$siblingU18)
value_2015_father = table(da_2015$fatherInHH)
value_2015_mother = table(da_2015$motherInHH)
value_2015_siblingU18 = table(da_2015$siblingU18)
value_2016_father = table(da_2016$fatherInHH)
value_2016_mother = table(da_2016$motherInHH)
value_2016_siblingU18 = table(da_2016$siblingU18)


####Father in HH####
plot_ly(
  labels = names(value_2017_father), 
  values = value_2017_father, 
  type = 'pie', 
  marker = list(colors = rainbow(length(value_2017_father))),
  textinfo = 'percent+label', 
  hoverinfo = 'label+value', 
  textposition = 'inside', 
  name = 'Father in HH'
) %>%
  layout(
    title = list(
      text = "Pie chart of father in HH for the year 2017 where mdeSI = NO",
      font = list(size = 14)
    )
)

plot_ly(
  labels = names(value_2015_father), 
  values = value_2015_father, 
  type = 'pie', 
  marker = list(colors = rainbow(length(value_2015_father))),
  textinfo = 'percent+label', 
  hoverinfo = 'label+value', 
  textposition = 'inside', 
  name = 'Father in HH'
) %>%
  layout(
    title = list(
      text = "Pie chart of father in HH for the year 2015 where mdeSI = Yes",
      font = list(size = 14)  # Adjust the font size as needed
    )
)

plot_ly(
  labels = names(value_2016_father), 
  values = value_2016_father, 
  type = 'pie', 
  marker = list(colors = rainbow(length(value_2016_father))),
  textinfo = 'percent+label', 
  hoverinfo = 'label+value', 
  textposition = 'inside', 
  name = 'Father in HH'
) %>%
  layout(
    title = list(
      text = "Pie chart of father in HH for the year 2016 where mdeSI = Yes",
      font = list(size = 14)  # Adjust the font size as needed
    )
)


#2017: father in HH: 72.2%
#2015: father in HH: 70.1%
#2016: father in HH: 68.0%

#Here we can see that presence of father in HH is not playing an important role in determining person suffering from depression

####mother in HH####
plot_ly(labels = names(value_2017_mother), values = value_2017_mother, type = 'pie', 
        marker = list(colors = rainbow(length(value_2017_mother))),
        textinfo = 'percent+label', 
        hoverinfo = 'label+value', 
        textposition = 'inside', 
        name = 'mother in HH') %>%
  layout(title = "Pie chart of mother in HH for the year 2017 whew mdeSI = No")

plot_ly(labels = names(value_2015_mother), values = value_2015_mother, type = 'pie', 
        marker = list(colors = rainbow(length(value_2015_mother))),
        textinfo = 'percent+label', 
        hoverinfo = 'label+value', 
        textposition = 'inside', 
        name = 'mother in HH') %>%
  layout(title = "Pie chart of mother in HH for the year 2015 where mdeSI = Yes")

plot_ly(labels = names(value_2016_mother), values = value_2016_mother, type = 'pie', 
        marker = list(colors = rainbow(length(value_2016_mother))),
        textinfo = 'percent+label', 
        hoverinfo = 'label+value', 
        textposition = 'inside', 
        name = 'mother in HH') %>%
  layout(title = "Pie chart of mother in HH for the year 2016 where mdeSI = Yes")


#2017: mother in HH: 91.0%
#2015: mother in HH: 91.2%
#2016: mother in HH: 89.5%

#Here we can see that presence of mother in HH is not playing an important role in determining person suffering from depression

####siblingU18####
plot_ly(labels = names(value_2017_siblingU18), values = value_2017_siblingU18, type = 'pie', 
        marker = list(colors = rainbow(length(value_2017_siblingU18))),
        textinfo = 'percent+label', 
        hoverinfo = 'label+value', 
        textposition = 'inside', 
        name = 'siblingU18') %>%
  layout(title = "Pie chart showing distribution of siblingU18 for the year 2017 where mdeSI = No")

plot_ly(labels = names(value_2015_siblingU18), values = value_2015_siblingU18, type = 'pie', 
        marker = list(colors = rainbow(length(value_2015_siblingU18))),
        textinfo = 'percent+label', 
        hoverinfo = 'label+value', 
        textposition = 'inside', 
        name = 'mother in HH') %>%
  layout(title = "Pie chart showing distribution of siblingU18 for the year 2015 where mdeSI = Yes")

plot_ly(labels = names(value_2016_siblingU18), values = value_2016_siblingU18, type = 'pie', 
        marker = list(colors = rainbow(length(value_2016_siblingU18))),
        textinfo = 'percent+label', 
        hoverinfo = 'label+value', 
        textposition = 'inside', 
        name = 'mother in HH') %>%
  layout(title = "Pie chart showing distribution of siblingU18 for the year 2016 where mdeSI = Yes")


#2017: siblingU18: 69.0%
#2015: siblingU18: 61.7%
#2016: siblingU18: 63.0%

#Here we can see that presence siblingU18 is not playing an important role in determining person suffering from depression

####ParentalInv####

# Calculate the percentage of each category for each year
da3_parentInv <- with(da3, prop.table(table(year, parentInv), margin = 1) * 100)

# Convert the data to a dataframe
da3_parentInv <- as.data.frame.matrix(da3_parentInv)

da3_parentInv$year <- rownames(da3_parentInv)

# Reset the row names to NULL
rownames(da3_parentInv) <- NULL

# Reshape the dataframe for plotting
da3_parentInv <- reshape2::melt(da3_parentInv, id.vars = "year")

p = ggplot(da3_parentInv, aes(x = year, y = value, fill = variable)) +
  geom_bar(stat = "identity") +
  labs(title = "Distribution of parentalInv across years",
       x = "Year",
       y = "Percentage") +
  scale_fill_manual(values = c("high parental involvement" = "green", "low parental involvement" = "red"), 
                    name = "Parental Involved") +
  theme_minimal()

ggplotly(p)

#Here we can notice that parental Involvement has increased significantly in 2017. So this can be considered as one of the factors
#for less depression cases in 2017.

####SchoolExp####

# Calculate the percentage of each category for each year
da3_schoolexp = with(da3, prop.table(table(year, schoolExp), margin = 1) * 100)

# Convert the data to a dataframe
da3_schoolexp = as.data.frame.matrix(da3_schoolexp)

da3_schoolexp$year = rownames(da3_schoolexp)

# Reset the row names to NULL
rownames(da3_schoolexp) = NULL

# Reshape the dataframe for plotting
da3_schoolexp = reshape2::melt(da3_schoolexp, id.vars = "year")

p = ggplot(da3_schoolexp, aes(x = year, y = value, fill = variable)) +
  geom_bar(stat = "identity") +
  labs(title = "Distribution of schoolExp across years",
       x = "Year",
       y = "Percentage") +
  scale_fill_manual(values = c("good school experiences" = "green", "bad school experiences" = "red"), 
                    name = "School Experience") +
  theme_minimal()

ggplotly(p)

#Here we can notice that good school experience has increased significantly in 2017. So this can be considered as one of the factors
#for less depression cases in 2017.

####Relation between income and mdeSI####
group_plot = as.data.frame(table(da$mdeSI, da$income,da$year))
colnames(group_plot) <- c("mdeSI", "income", "year", "count")
plot = ggplot(group_plot, aes(x = year, y = count, fill = mdeSI)) +
  geom_bar(stat = "identity", position = "stack") +
  facet_wrap(~ income, ncol = 2) +  # Group by income and create separate facets
  labs(title = "Distribution of mdeSI by income and year",
       x = "Year",
       y = "Count",
       fill = "mdeSI") +
  theme_minimal()
ggplotly(plot)

#One interesting finding from this is that as the income is increasing, number of mdeSI cases are also increasing.
#This needs to be looked into why increase in income is leading to more mdeSI cases.
#Thus, income becomes one important variable for the classification model

####Relation between gender and mdeSI####
#First we will see the distribution of the variable gender
table(da$gender)
#From this we can see that there are total 3547 females and 2453 males

#now we will see which gender is suffering from mdeSI the most
mdeSI_yes = subset(da, mdeSI == "Yes")
gender_dist = as.data.frame(table(mdeSI_yes$gender))
colnames(gender_dist) <- c("Gender", "Count") #Renaming the columns
plot_ly(gender_dist, labels = ~Gender, values = ~Count, type = "pie") %>%
  layout(title = "Distribution of Gender for mdeSI = Yes")

mdeSI_no = subset(da, mdeSI == "No")
gender_dist2 = as.data.frame(table(mdeSI_no$gender))
colnames(gender_dist2) <- c("Gender", "Count") #Renaming the columns
plot_ly(gender_dist2, labels = ~Gender, values = ~Count, type = "pie") %>%
  layout(title = "Distribution of Gender for mdeSI = No")

#From the two pie chart, we can see that 75% of the females(2249) are suffering from mdeSI compared to male which is 25%(751)
#So finding out the reason why more females are havind mdeSI problems is important. This implies gender is an important variable
#for classification model

####Race and mdeSI####
race_mdeSI_counts = table(da$race, da$mdeSI)

# Calculate the percentages of mdeSI = "Yes" for each race
percent_yes = race_mdeSI_counts[, "Yes"] / rowSums(race_mdeSI_counts) * 100

# Create a dataframe with race and percentage of mdeSI = "Yes"
race_percent_yes = data.frame(race = rownames(race_mdeSI_counts), percent_yes = percent_yes)

race = ggplot(race_percent_yes, aes(x = race, y = percent_yes)) +
  geom_bar(stat = "identity", fill = "skyblue", width = 0.5) +
  labs(title = "Percentage of mdeSI = 'Yes' for Each Race",
       x = "Race",
       y = "Percentage of mdeSI = 'Yes'",
       fill = "Race") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggplotly(race)

#We can see that the mdeSI cases are almost in the same interval for all race. Hence, it cannot be concluded that race is an
#impotant variable to determine mdeSI.

####age and mdeSI####
age_mdeSI_counts = table(da$age, da$mdeSI)

# Calculate the percentages of mdeSI = "Yes" for each race
percent_yes = age_mdeSI_counts[, "Yes"] / rowSums(age_mdeSI_counts) * 100

# Create a dataframe with race and percentage of mdeSI = "Yes"
age_percent_yes = data.frame(age = rownames(age_mdeSI_counts), percent_yes = percent_yes)

age = ggplot(age_percent_yes, aes(x = age, y = percent_yes)) +
  geom_bar(stat = "identity", fill = "skyblue", width = 0.5) +
  labs(title = "Percentage of mdeSI = 'Yes' for Each age group",
       x = "age",
       y = "Percentage of mdeSI = 'Yes'",
       fill = "age") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggplotly(age)

#We can see that increase in age is leading to more cases of mdeSI. Hence, this variable needs attention.

####Finding relation amongst the columns####

#Chi Square test
chi_square_results = data.frame(Column = character(), P_Value = numeric(), row.names = NULL)

columns = c("year", "gender", "age", "race", "insurance", "income", 
             "fatherInHH", "motherInHH", "siblingU18", "parentInv", "schoolExp")

for (col in columns) {
  chi_square_result = chisq.test(table(da$mdeSI, da[, col]))
  p_value = round(chi_square_result$p.value,4)
  chi_square_results = rbind(chi_square_results, data.frame(Column = col, P_Value = p_value))
}
chi_square_results

#From the chisq test, we can conclude that variables: year, gender, age, race, income, father in HH, siblingU18, parent Inv and schoolExp are significant to our target variable

da2 = da
da2$mdeSI <- ifelse(da2$mdeSI == "Yes", 1, 0)
da2$gender = as.numeric(unclass(da2$gender))
da2$age = as.numeric(unclass(da2$age))
da2$race = as.numeric(unclass(da2$race))
da2$insurance = as.numeric(unclass(da2$insurance))
da2$income = as.numeric(unclass(da2$income))
da2$fatherInHH = as.numeric(unclass(da2$fatherInHH))
da2$motherInHH = as.numeric(unclass(da2$motherInHH))
da2$siblingU18 = as.numeric(unclass(da2$siblingU18))
da2$parentInv = as.numeric(unclass(da2$parentInv))
da2$schoolExp = as.numeric(unclass(da2$schoolExp))


mcor = round(cor(da2),2) #Finding pair wise correlation
corrplot(mcor, type = "lower")
#From the correlation plot, we can see that variables: year, gender, age, parentInv and schoolExp are correlating the most with
#Our target variable

#Thus we will make 2 regression model based on chisq test and correlation plot and see which model gives better accuracy
#Logistic regression model is based on chisq test and random forest based on correlation plot and data visualization

#Build your two classifier:

####Insurance and mdeSI####
table(da$mdeSI, da$insurance)
#we can see that there is a similar distribution of mdeSI (Yes/No) across both categories of insurance (Yes/No)
#Thus we can say that insurance is not playing a major role towards the target variable mdeSI.
####(A) Classifier one - Logistic Classifier####

#Training model
model1 = glm(mdeSI ~ year + gender + age + income + fatherInHH + siblingU18 + parentInv + schoolExp, family = binomial(logit),
                   data = train)
summary(model1)
anova(model1)
prob01 = predict(model1, train, type = "response")
pred01 = rep(0, 4500)
pred01[prob01 > 0.5] = 1
(CM01 = table(pred01,train$mdeSI)) #confusion matrix
(accuracy01 = (CM01[1,1] + CM01[2,2])/4500)
(recall01 = CM01[2,2]/sum(CM01[,2]))
(precision01 = CM01[2,2]/sum(CM01[2,]))

#Testing Model
prob01.test = predict(model1, test, type = "response")
pred01.test = rep(0, 1500)
pred01.test[prob01.test > 0.5] = 1
(CM01.test = table(pred01.test,test$mdeSI)) #confusion matrix
(accuracy01.test = (CM01.test[1,1] + CM01.test[2,2])/1500)
(recall01.test = CM01.test[2,2]/sum(CM01.test[,2]))
(precision01.test = CM01.test[2,2]/sum(CM01.test[,2]))

#Now we will plot accuracy and recall of both training and testing data
accuracy_recall = data.frame(Model = c("Training", "Testing"),
                              Accuracy = c(accuracy01, accuracy01.test),
                              Recall = c(recall01, recall01.test))

acc_rec_plot = ggplot(accuracy_recall, aes(x = Model)) +
  geom_bar(aes(y = Accuracy, fill = "Accuracy"), stat = "identity", position = position_dodge(width = 0.8)) +
  geom_bar(aes(y = Recall, fill = "Recall"), stat = "identity", position = position_dodge(width = 0.8)) +
  labs(title = "Accuracy and Recall of Training and Testing Models",
       x = "Model", y = "Value", fill = "Metric") +
  theme_minimal()
ggplotly(acc_rec_plot)

#AUC value and ROC plot
roc_train = roc(train$mdeSI, prob01, levels = c("No", "Yes"))
(auc_train = auc(roc_train))

roc_test = roc(test$mdeSI, prob01.test, levels = c("No", "Yes"))
(auc_test = auc(roc_test))

plot(roc_train, col = "blue", main = "ROC Curve for Training and Testing Models")
lines(roc_test, col = "green")
legend("bottomright", legend = c("Train", "Test"), col = c("blue", "green"), lty = 1)

####(B) Classifier two - Random Forest####

#Finding the best ntree value
#Partitioning training data into train and validation data
set.seed(2024)
trainIndex = createDataPartition(train$mdeSI, p = 0.8, list = FALSE)
train_data = train[trainIndex, ]
validation_data = train[-trainIndex, ]
ntree_values = c(50, 100, 200, 300, 500)
cv_results = lapply(ntree_values, function(ntree) {
  model = randomForest(mdeSI ~ year + gender + age + income + parentInv + schoolExp,
                        data = train_data, ntree = ntree)
  pred = predict(model, newdata = validation_data)
  accuracy = mean(pred == validation_data$mdeSI)
  return(list(ntree = ntree, accuracy = accuracy))
})

#looking at the accuracy for different values of ntree and finding it's best value based on cross-validation results
(cv_results_df = do.call(rbind, cv_results))

#We can see that for ntree value of 300 & 500, the accuracy is the same. Hence we will go with ntree value of 300 for time optimization

#Training model
rf = randomForest(mdeSI ~ year + gender + age + income + parentInv + schoolExp, data = train, ntree = 300)
rf_preds <- predict(rf, train)
(rf_table = table(rf_preds, train$mdeSI))
(rf_accuracy = sum(diag(rf_table))/sum(rf_table))
(rf_recall = rf_table[2,2]/sum(rf_table[,2]))

#Testing Model
rf_preds.test <- predict(rf, test)
(rf_table.test = table(rf_preds.test, test$mdeSI))
(rf_accuracy.test = sum(diag(rf_table.test))/sum(rf_table.test))
(rf_recall.test = rf_table.test[2,2]/sum(rf_table.test[,2]))

#Accuracy and recall plot
accuracy_recall_rf = data.frame(Model = c("Training", "Testing"),
                                Accuracy = c(rf_accuracy, rf_accuracy.test),
                                Recall = c(rf_recall, rf_recall.test))
acc_rec_plot_rf = ggplot(accuracy_recall_rf, aes(x = Model)) +
  geom_bar(aes(y = Accuracy, fill = "Accuracy"), stat = "identity", position = position_dodge(width = 0.8)) +
  geom_bar(aes(y = Recall, fill = "Recall"), stat = "identity", position = position_dodge(width = 0.8)) +
  labs(title = "Accuracy and Recall of Train and Test Models",
       x = "Model", y = "Value", fill = "Metric") +
  theme_minimal()
ggplotly(acc_rec_plot_rf)


#AUC Value and ROC plot
rf_probs_train = predict(rf, train, type = "prob")[, "Yes"]
rf_probs_test = predict(rf.test, test, type = "prob")[, "Yes"]

roc_train_rf = roc(train$mdeSI, rf_probs_train, levels = c("No", "Yes"))
(auc_train = auc(roc_train_rf))

roc_test_rf = roc(test$mdeSI, rf_probs_test, levels = c("No", "Yes"))
(auc_test = auc(roc_test_rf))

plot(roc_train_rf, col = "red", main = "ROC Curve for Train and Test Models")
lines(roc_test_rf, col = "purple")
legend("bottomright", legend = c("Train", "Test"), col = c("red", "purple"), lty = 1)

