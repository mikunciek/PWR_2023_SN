import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Load and prepare database
df = pd.read_csv('titanic.csv')
df = df.drop(['sibsp','ticket','fare','boat','body','cabin'], axis=1) #removing unnecessary columns
df[df.duplicated()==True] #we check if we have duplicate records
df.isnull().sum()  #lack ages, ports

#How many people survided and how many died?
sns.set_palette("Set1")
sns.catplot(data=df,x='survived', kind='count')
plt.show()

#Did the class people were traveling in have an impact on survival?
# Travelers who traveled first class had a better chance of survival. Either they were evacuated first, or their cabins were located closer to the lifeboats
sns.catplot(data=df,x='pclass', kind = 'count',hue= 'survived')
plt.show()
#Did the person's age affect survival? Most of the children survived
sns.distplot(df[df['age'].notnull() & (df['survived']==1)]['age'],kde_kws={"label": "Survived"},bins=10)
sns.distplot(df[df['age'].notnull() & (df['survived']==0)]['age'],kde_kws={"label": "Not Survived"},bins=10)
plt.show()

#Did the person's gender influence survival?
# Yes, women had a better chance of survival
sns.catplot(data=df, x = 'sex',hue = 'survived',kind='count')
plt.show()



