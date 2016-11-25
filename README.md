# simple-machine-learning
try some simple machine learning and data science.
## p1
Implement the common k-means algorithm on dataset named realdata, which is in the package. Plot the figure.
## p2
Implement the common logistic regression on dataset with arff format and the information about dataset is placed in the package also.<br>
Run different values of logistic regression regularization parameter (λ). The range of λ is from -2 to 4 and the step is 0.2.<br>
Plot the f-measure of the algorithm’s performance on the training set as a function of:<br>
f-measure= 2 × Pr × Re/(Pr + Re) where: Pre= TP/(TP + FP); Rec= TP/(TP + FN);
- TP is the number of true positives (class 1 members predicted as class 1),
- TN is the number of true negatives (class 2 members predicted as class 2),
- FP is the number of false positives (class 2 members predicted as class 1),
- FN is the number of false negatives (class 1 members predicted as class 2).<br>
Besides, use z-score standardization and do it again.


## p3
Apply three clustering techniques to the handwritten digits dataset. Assume with k=10:
- k-means clustering implemented above
- Agglomerative clustering with Ward linkage
- AffinityPropagation<br>
The primary dataset is the handwritten digits datasets.load_digits with description available here:<br>
http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits<br>
Report the confusion matrix.<br>
Calculate the accuracy of each clustering method using Fowlkes and Mallows index:<br>
https://en.wikipedia.org/wiki/Fowlkes–Mallows_index


## p4
Apply three classification techniques to the same realdata1.zip dataset as in p2:
- LinearSVC
- RbfSVC
- Randomforest<br>
The mearsure rule is same with p2.
