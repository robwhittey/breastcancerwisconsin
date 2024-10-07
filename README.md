# breastcancerwisconsin
Machine learning









<h2>§1: Introduction </h2>

In this report, we will be looking at the classification of breast cancer diagnoses. We will be using a variety of machine learning systems and one deep learning model with the aim of finding the best system for classification. For each of the systems, we will be fine tuning the associated hyperparameters with the objective of obtaining the best prediction level that model can do. Beforehand, we will look at how pre-processing the data influences prediction scores and what training-testing proportion is best for our analysis.
<h2>§2: Data Overview</h2>

The data we will be using is a dataset of breast cancer diagnoses with their feature measurements. These features are extracted from fine needle aspirations of breast tissue mass and the dataset includes mean, standard error, and greatest values of 10 feature measurements of a cell nucleus. 

There are a total of 569 measurements in the dataset with each having a total of 30 features and a corresponding diagnosis. (M: Malignant, B: Benign). There are 357 ‘Benign’ and 212 ‘Malignant’ diagnoses. This is considered an unbalanced dataset which may hinder results as the algorithms may favour the most occurring. 


Figure 1: Bar graph of the number of each diagnosis (212 Malignant and 357 Benign).

Finally, we will look to see if there are any correlations between the variables. The machine learning system will be able to analyse this in multidimensions but for now, we will look at the data in 2 dimensional and observe. 

Figure 2: Scatter pair plot of all the variables

In the scatter plots above (see figure 2), we can see a number of graphs with a correlation between the two variables. This is confirmed with the heatmap below (see figure 3) which gives a numerical value on the correlation level of each variable against one another. This bodes well for our analysis, and we should expect high accuracy for some of our models. Although, correlations found here may not be useful and have no relevance to our classification. 


Figure 3: Heatmap of regression levels found in dataset
<h2>§3: Discussion of the problem</h2>

When it comes to machine learning for a classification problem such as this one, there are a number of options and avenues that would work and produce acceptable results. We may be able to produce a system which has an accuracy greater than 96% from the dataset, but with the dataset being relatively small (Ajiboye et al., 2015), our model may not be adequate as a sole means of diagnosis. But instead, will be viable as a proof of concept.

Breast cancer, like many other diseases, are best caught early and our number one goal for this model is not just to increase the accuracy, but to reduce the number of False Negatives (FN) and False Positives (FP). Specifically, reducing the number of False Negatives is more important as we want to reduce the number of patients who have undiagnosed cancer. We will therefore focus on the accuracy of the model, the precision of the model, and the recall. The best system will be the one which has the greatest of each of these scores.


<h2>§4: Training </h2>

In this section, we will be looking at six different machine learning models and fine tuning them to produce the best of each, ready for comparison. But first, we need to separate the data into a training and testing dataset and find the best setup for the data. 

Initial testing has found, using the Logistic Regression (see §4.1 Logistic Regression) as a test, that normalising the data improved all metric variables (Accuracy, Precision and Recall) with an increase of 2% for accuracy and precision by over 7%. The random state at which we select the data for splitting, when changed to 42, again, further increased the accuracy by 1% and increased the recall to 98%. After comparing the train-test ratio, we find that a 60%-40% split performed the best whilst also being within the bounds of convention (Joseph, 2022). 



§4.1: Logistic Regression

The logistic regression model works by finding linear combinations of the features (Edgar and Manz, 2017). The model has many solvers and one can change the maximum number of iterations to allow convergence. In testing, when using the natural data that had not been normalised, we see differences when we use different solvers and higher iterations. But when we use normalised data, all solvers and iteration levels (above 500) produce the same metric results and confusion matrix.


Figure 4: Logistic regression with ‘newton-cg’ solver. Natural data (left) and normalised data (right).

<h3>§4.2: Genetic Algorithm</h3>

Genetic Algorithms are based on the Theory of Evolution and take random mutations and natural selection into account to produce the ‘fittest’ model. It begins with a Logistic Regression model with variables such as number of generations, number of parents and mutation rate. Setting these variables to 20 generation, 10 parents and a mutation rate of 0.01, produces an initial accuracy of 0.982 (3.s.f), precision of 0.963 (3.s.f), and recall of 0.988 (3.s.f). The accuracy and precision improve over generations with a final accuracy score of 0.991 and precision of 1.00. Although, recall did not improve and stayed the same throughout.

<h3>§4.3: Decision Tree</h3>

Decision trees use conditional nodes at branching points which lead to ‘leaves’ as an output. The depth of a decision tree can be set beforehand but for this test, we will compare the original tree with a pruned tree. 


Figure 5: Decision tree evaluation with original (left) and pruned (right) using ccp_alpha=0.005.

As we can see from the evaluation above (figure 4), the pruned tree produced a much better model with a 4% increase in accuracy and precision. Unfortunately, the recall only increased by 1%. Below (see figure 5), is an insight into how the model works and decides on its classification. 

Figure 6: Pruned decision tree plot of how the model classifies.

<h3>§4.4: Random Forest</h3>

Random forests are made of many decision trees (see §4.3: Decision tree) which are made at random. The classification is decided by the most popular classification across the whole forest of trees. Unexpectedly, the metrics for this model supersede that of the decision tree with a 2% gain in accuracy, 6% gain in precision and 1% gain in recall.


Figure 7: Random forest evaluation.

<h3>§4.5: Support Vector Machine</h3>

The Support Vector Machine (SVM) allows the use of different kernels to aid classification with regression. We will look at the linear, polynomial, radial basis function (RBF), and sigmoid kernel. For some of these kernels, we were able to tune the hyperparameters of the function to produce higher metric values. The results of each are below (see figure 7). The best by far was the linear kernel with a C value of 0.2.


 
Figure 8: SVM with linear (top left), polynomial (top right), RBF (bottom left) and
 sigmoid (bottom right) kernels.


<h3>§4.6: ANN – deep learning</h3>

To further our analysis, we will move away from machine learning systems and use a simple deep learning model. Artificial Neural Networks (ANN) are based on the neural networks found in the brain with inter-connecting neurons with inputs and outputs and associated weights. ANN use hyperparameters such as learning rates and epochs to help train the model. The model sends batches of data back and forth through the network, adjusting weights as it does so to improve the accuracy of the prediction. Below (see figure 8), is the architecture of the ANN with an input layer, a single hidden layer, and an output layer. 


Figure 9: Architecture of ANN model with 3 layers

Setting the epoch level to 1000, the model was able to achieve an accuracy of 97.95% and a loss of 0.1377 for the training data. When using the test, we gained an accuracy of 96.9%, a precision of 92.9%, and a recall of 98.8%. 


Figure 10: Output evaluation of ANN model and confusion matrix.




<h2>§5: Results</h2>

After training the models and evaluating each one, we can now compare the results of the metric scores and decide on the best model. As we can see from the table below (see figure 10), overall, the genetic algorithm performed the best throughout the metric scores. 


Figure 11: Table comparing the metrics of each model

<h2>§6: Conclusion</h2>

In §3: Discussion of the problem, we highlighted the importance of reducing the level of False Negative predictions which meant we wanted a higher recall score for classification. The genetic algorithm came out on top with all metrics and as for recall, we found logistic regression and ANN’s also produced the same recall score as genetic algorithms but with lower accuracy and precision (see figure 10). 

However, for a system such as this to be fully utilised in the medical field, an element of transparency and explanation is required, if not, essential. Genetic algorithms and ANN are both systems which are considered black box; the reasoning behind the classification is hidden and unexplainable (so far). It is therefore my conclusion to further development and fine tuning of logistic regression and support vector machine. With these models, we are able to get a hint at how the model performs classification using SHAP, PDP or LIME to get a breakdown on how variables influence the classification. Perhaps with a more refined dataset with the removal of some datapoints would increase the accuracy and precision (maybe even the recall) too. 






<h2>References</h2>

Ajiboye, A.R., Abdullah-Arshah, R., Qin, H. and Isah-Kebbe, H. (2015). EVALUATING THE EFFECT OF DATASET SIZE ON PREDICTIVE MODEL USING SUPERVISED LEARNING TECHNIQUE. International Journal of Computer Systems & Software Engineering, 1(1), pp.75–84. doi:10.15282/ijsecs.1.2015.6.0006.
Edgar, T.W. and Manz, D.O. (2017). Logistic Regression - an overview | ScienceDirect Topics. [online] www.sciencedirect.com. Available at: https://www.sciencedirect.com/topics/computer-science/logistic-regression.
Joseph, V.R. (2022). Optimal ratio for data splitting. Statistical Analysis and Data Mining: The ASA Data Science Journal, 15(4), pp.531–538. doi:10.1002/sam.11583.

