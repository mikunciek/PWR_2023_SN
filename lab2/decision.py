import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pandas.read_csv("data.csv") #Load dataset

#Converting values to numbers (Decision Tree Requirement)
d = {'UK': 0, 'USA': 1, 'N': 2} # Means convert the values 'UK' to 0, 'USA' to 1, and 'N' to 2.
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)

#Separation of the feature column (x) from the target column (y)
features = ['Age', 'Experience', 'Rank', 'Nationality', 'Go']

array = df.values
# The feature columns are the columns that we try to predict from, and the target column is the column with the values we try to predict.
X = array[:, :-1]
Y = array[:, -1]

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, Y)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(dtree,
                   feature_names=df.columns,
                   class_names=features,
                   filled=True)
fig.savefig("decision_tree_2.png")


#Use predict() method to predict new values - #What would the answer be if the comedy rank was 6?
print( f"Predict:  {dtree.predict([[40, 10, 6, 1]])}")

print("[1] means 'GO'")
print("[0] means 'NO'")








