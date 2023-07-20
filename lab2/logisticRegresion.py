import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
#data
x = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

#model
model = LogisticRegression(solver='liblinear', random_state=0).fit(x, y)

#attributes of model
cl = model.classes_
inter = model.intercept_
coef = model.coef_
#evaluate model
model.predict_proba(x)
model.predict(x)
score = model.score(x, y)
cm = confusion_matrix(y, model.predict(x))

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

#print values
print(f"binary classification: {cl}")
print(f"intercept: {inter}") #b0
print(f"slope: {coef}") #b1
print(f"predict: {model.predict(x)}") #predict
print(f"accuracy: {score}")
print(f"confusion matrix: {cm}")
print(f"More comprehensive report on the classification: {classification_report(y, model.predict(x))}")#more comprehensive report on the classification





