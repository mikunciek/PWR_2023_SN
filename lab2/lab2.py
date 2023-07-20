import pandas
from sklearn.metrics import classification_report, confusion_matrix
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
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

#Logistic Regression#--------------------------------------------
modelLR = models[0][1].fit(X, Y) #model
cl = modelLR.classes_
inter = modelLR.intercept_
coef = modelLR.coef_
#evaluate model
modelLR.predict_proba(X)
modelLR.predict(X)
score = modelLR.score(X, Y)
cm = confusion_matrix(Y, modelLR.predict(X))
#heatmap
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()
print('LOGISTIC REGRESSION ---------------------------------------')
print(f"binary classification: {cl}")
print(f"intercept: {inter}") #b0
print(f"slope: {coef}") #b1
print(f"predict: {modelLR.predict(X)}") #predict
print(f"accuracy: {score:f}")
print(f"confusion matrix: {cm}")
print(f"More comprehensive report on the classification: {classification_report(Y, modelLR.predict(X))}")#more comprehensive report on the classification
#Decision Tree #---------------------------------------
modelDT = models[1][1].fit(X, Y)
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(modelDT,
                   feature_names=dataset.columns,
                   class_names=names,
                   filled=True)
fig.savefig("decision_tree.png")

#Evaluate each model in turn -------------------------------------
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





