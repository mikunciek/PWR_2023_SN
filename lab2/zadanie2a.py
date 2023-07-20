import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
#Load dataset
url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names=["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
dataset = pandas.read_csv(url, names=names)

dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False) #Box and whisher plots
plt.show()
dataset.hist() #Histograms
plt.show()
scatter_matrix(dataset) #Scatter plot matrix
plt.show()

#Split-out validation dataset
array = dataset.values
X = array[:, :-1]
Y = array[:, -1]
validation_size =0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
scoring = 'accuracy'

#Spot Check Algorithms
models =[]
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr'))) #Logistic Regression
models.append(('CART', DecisionTreeClassifier())) #Decision Tree

#Evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg ="%s: %f (%f)" % (name, cv_results.mean(),cv_results.std())
    print(msg)
#Compare Algorithms
fig =plt.figure()
fig.suptitle('Algorithm Comparison')
ax =fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
