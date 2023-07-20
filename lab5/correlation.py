import pandas
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from matplotlib import pyplot

#Load dataset
url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names=["sepal-length", "sepal-width", "petal-length", "petal-width", "class"]
dataset = pandas.read_csv(url, names=names)
#data
data1 = dataset.loc[:, 'sepal-length']
data2 = dataset.loc[:, 'petal-length']

#show data
pyplot.scatter(data1, data2)
pyplot.show()

# Pearson's Correlation test
print("Pearson's correlation")
stat, p = pearsonr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
 print('Probably independent')
else:
 print('Probably dependent')

#Spearman's Rank Correlation Test
print(' ')
print("Spearman's correlation")
stat, p = spearmanr(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
 print('Probably independent')
else:
 print('Probably dependent')

#Kendall's Rank Correlation Test
print(' ')
print("Kendall's correlation")
stat, p = kendalltau(data1, data2)
print('stat=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
 print('Probably independent')
else:
 print('Probably dependent')








