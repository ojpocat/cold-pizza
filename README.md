# Happiness & Social Media Usage Analysis

## Introduction

Our project focuses on social media usage and how it can affect happiness and mental health. The rapid growth of social media platforms has significantly transformed the way individuals communicate, share information, and interact with the world. As daily screen time and online engagement continue to rise, concerns have emerged about the potential impact of social media usage on mental health and overall happiness. The increasing use of social media has been identified as contributing to an increase in mental health challenges. [1] Analyzing trends between happiness and social media usage informs healthier platform design and better strategies for safe, balanced use. In this project, we apply machine learning methods to analyze variables correlated with social media usage and examine how they relate to a quantified happiness score. Our goal is to better understand how different aspects of social media engagement may positively or negatively affect mental health. Insights gained from this analysis can help inform healthier platform design and support strategies that encourage balanced, mindful social media use.

## Data

The dataset for this analysis comes from Kaggle as tidy data. [2] Each row represents an individual, and the columns represent their lifestyle and media habits. The variables used in the analysis are _Daily Screen Time_, _Stress Level_, and _Sleep Quality_. We selected these variables for the analysis because our exploratory data analysis showed us that these variables have the highest correlation to our happiness index. The outcome variable that we are using is _Happiness Index_, a numerical variable dictating the overall happiness of the individual based on a self-reported score. We also created a scaled, binned version of _Happiness Index_ which places each score into either Low, Medium, or High.

<table><tr><td><img src="plots/positive_correlations.png"></td><td><img src="plots/negative_correlations.png"></td></tr></table>

## Methods

We are using several machine learning techniques using variables from the data on our happiness index variable. We have analysis on the happiness variable as numerical and as a binned variable. 

Our first analysis is on happiness index as a numerical variable. The models that we tested were **Linear Regression**, 

// We ran all the models on the binned and numerical variable, and then decided based on that which model worked best for each case. 

## Results

The best model overall is KNN, and the binned models performed better than the numerical models. 

## Learnings

## Conclusion

We can see the that the most effective models for this data are 

## References

1. Beyari, H. (2023). The Relationship between Social Media and the Increase in Mental Health Problems. International Journal of Environmental Research and Public Health, 20(3), 2383. https://doi.org/10.3390/ijerph20032383

2. Kaggle dataset
