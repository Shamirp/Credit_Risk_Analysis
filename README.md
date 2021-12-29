# Credit_Risk_Analysis

## Overview of the analysis

Apply machine learning to solve a real-world challenge: credit card risk.
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, we will employ different techniques to train and evaluate models with unbalanced classes using imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

## Results

### Naive Random Oversampling
![NROversampling](https://user-images.githubusercontent.com/88383836/147683819-63231d8b-95d9-4514-81f5-37ef9d768ef7.PNG)

•	Naïve Random Oversampling: According to the confusion matrix the high-risk precision score is 1% (70/(70+6933) = .0099) which is a very low positivity rate. This concludes that most loans are of the low-risk variety.


### SMOTE Oversampling
![SMOTE](https://user-images.githubusercontent.com/88383836/147683960-3c9e2aaf-1023-4654-83a0-48233e48c994.PNG)


•	SMOTE Oversampling:  The results are similar to the Naïve Random Oversampling results. The SMOTE method, however, provides a slightly higher balanced accuracy score at 66% and a lower high risk recall percentage (63%). This leaves slightly more ambiguity to the data outputs and may call for more concern in the correlation between the relationship of the values. 


### Undersampling
![Undersampling](https://user-images.githubusercontent.com/88383836/147684166-6485edaf-4259-4238-8fe0-8d4ea6c55707.PNG)

•	Undersampling: The results show a higher high risk recall value at 69% and the higher balance accuracy score at 66%. The low-risk recall score is 40% which is the lowest of all the sample tests. This confirms a low correlation between the relationship of low-risk loans compared to the parameters we are measuring. 


### Combination
![Combination](https://user-images.githubusercontent.com/88383836/147684382-87a243bb-95cd-44b8-aad3-c31d5e7de132.PNG)

•	Combination (Over and Under Sampling): The results from the combination sample show a lower balance accuracy score at 64% with the highest high risk recall percentage at 70%. 


### Balanced Random Forest Classifier 
![BalancedRF](https://user-images.githubusercontent.com/88383836/147684501-c39c2a23-837c-4cf2-ae2a-e8be9089b49f.PNG)

•	Balanced Random Forest Classifier: This set of results yield a significantly higher balanced accuracy score at 80% and very high recall percentages for both high and low risk loans. This gives a stronger sense of the true correlation between the datasets. 


###	Easy Ensemble AdaBoost Classifier
![EEC](https://user-images.githubusercontent.com/88383836/147684606-41fd6f21-15bb-4e36-8213-39c11f703906.PNG)

•	Easy Ensemble AdaBoost Classifier: Has the lowest recall percentage for high-risk loans at 11%. 


## Summary 

The first four sampling models provided some of the lowest balanced accuracy scores making it hard to consider using these models as a method for best fit when finding a correlation in our dataset. The Balanced Random Forest Classifier model yielded the best results for the balance accuracy score (80%) and the best results for recall percentages in high and low risk loans which gives strong support for the correlation being accurately sampled. My recommendation would be to not use the Easy Ensemble AdaBoost Classifier. In terms of compatibility with the kernel it requires additional measures to get results. The installation of a previous version of scikit-learn is needed to provide feedback for the analysis whereas all the other samples are compatible with the current mlenv kernel configuration. This can cause setbacks and require additional time to import and train the dataset. 




