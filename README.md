# Happiness & Social Media Usage Analysis

## Introduction

Our project focuses on social media usage and how it can affect happiness and mental health. The rapid growth of social media platforms has significantly transformed the way individuals communicate, share information, and interact with the world. As daily screen time and online engagement continue to rise, concerns have emerged about the potential impact of social media usage on mental health and overall happiness. The increasing use of social media has been identified as contributing to an increase in mental health challenges. [1] Analyzing trends between happiness and social media usage informs healthier platform design and better strategies for safe, balanced use. In this project, we apply machine learning methods to analyze variables correlated with social media usage and examine how they relate to a quantified happiness score. Our goal is to better understand how different aspects of lifestyle habits related to social media can affect mental health. We are trying to see if we can use these lifestyle factors to predict someone's mental well-being and how effective our model can make the predictions. We will then see if we can determine certain behaviours mose indicative of mental well-being. 

## Data

The dataset for this analysis comes from Kaggle as tidy data. [2] Each row represents an individual, and the columns represent their lifestyle and media habits. The variables used in the analysis are _Daily Screen Time_, _Stress Level_, and _Sleep Quality_. We selected these variables for the analysis because our exploratory data analysis showed us that these variables have the highest correlation to our happiness index. The outcome variable that we are using is _Happiness Index_, a numerical variable dictating the overall happiness of the individual, and representing well-being, based on a self-reported score. We also created a scaled, binned version of _Happiness Index_ which places each score into either Low, Medium, or High.

<table><tr><td><img src="plots/positive_correlations.png"></td><td><img src="plots/negative_correlations.png"></td></tr></table>

## Methods

We are using several machine learning techniques using variables from the data on our happiness index variable. We have analysis on the happiness variable as numerical and as a binned variable. The first part of our analysis is on happiness index as a binned variable. The models that we tested were classification models **Logistic Regression**, **Random Forest**, **SVM**, and **KNN**. The second part of our analysis is on happiness index as a numerical variable. The models that we tested were regression models **Linear Regression**, **Random Forest**, **SVM**, and **KNN**. Then we categorized our predicted happiness indeces from the regression models into the happiness bins to compute an accuracy and compare with the classification models.

Because we determined the 3 variables, our models determined which variable was the most influential predictor variable. 

## Results

The results of our analysis indicate that the classified K-Nearest Neighbors (KNN) model performed best overall. Across all evaluated metrics, classification-based models consistently outperformed their regression counterparts. This suggests that predicting well-being as a categorical outcome (e.g., low, medium, high) is more effective than attempting to predict it on a continuous scale.

While classification models performed better overall, differences within each model type were relatively small. For example, among classification models, KNN and Random Forest achieved similar performance, and a similar pattern was observed among regression models. This indicates that model choice within the same modeling framework was less impactful than the decision between classification and regression itself.

Overall, stress level was the most important predictor variable based on the results of the feature importance across models. 

![Results Comparing Models](Results_graph.png)

## Learnings

This project demonstrated that comparing regression and classification models can be challenging, as they optimize different objectives and rely on distinct evaluation metrics. Direct performance comparisons are therefore not always straightforward, particularly when the target variable can reasonably be framed in multiple ways.

The strong performance of KNN and Random Forest models suggests that the underlying relationships between predictors and well-being are likely non-linear. Although some feature relationships may appear linear at a surface level, models that capture local patterns (KNN) or complex interactions (Random Forest) were better suited to the task.

Additionally, this project highlighted the inherent difficulty of predicting subjective outcomes such as well-being or happiness.

## Conclusion

The goal of this project was to use machine learning methods to examine how lifestyle factors related to social media usage are associated with mental well-being, and to evaluate how effectively these factors can be used to predict a happiness index indicating mental well-being. Through this analysis, we found that prediction accuracy was strongly influenced by how the target variable was defined. Binning the happiness score into categories resulted in more reliable predictions than modeling it as a continuous numerical value, suggesting that mental well-being may be better captured at a categorical level rather than as an exact score.

Our results show that no single model was optimal for all objectives. KNN performed best when predicting well-being categories, while Random Forest produced stronger results for numerical prediction tasks. This supports our broader goal of understanding model effectiveness, where the choice of algorithm should be guided by the specific prediction goal rather than overall performance alone. Additionally, differences in feature importance across models highlight that no single behavior can be declared the most important factor for mental health.

In our analysis, the variables stress level, sleep quality, and daily screen time were the most significant predictors of mental well-being, aligning with the project’s aim of identifying behaviors most indicative of mental health. However, the varying importance of these features across models indicates that mental well-being is shaped by nonlinear, interacting lifestyle factors, not isolated habits.

Although our project was able to indentify some patterns in the prediction of happiness index, it emphasizes the limitations of predictive models when applied to subjective outcomes like happiness. While our models can support understanding and generate insights, they cannot fully capture or measure true mental health, showing the limits of data-driven prediction.

## References

1. Beyari, H. (2023). The Relationship between Social Media and the Increase in Mental Health Problems. International Journal of Environmental Research and Public Health, 20(3), 2383. https://doi.org/10.3390/ijerph20032383

2. Rajak, P. (2025, October 15). Mental Health & Social Media Balance Dataset. Kaggle. https://www.kaggle.com/datasets/prince7489/mental-health-and-social-media-balance-dataset/data 

3. ChatGPT
