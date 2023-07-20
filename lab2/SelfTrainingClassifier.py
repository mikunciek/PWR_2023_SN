import pandas as pd
from sklearn.model_selection import train_test_split # for splitting data into train and test samples
from sklearn.svm import SVC # for Support Vector Classification baseline model
from sklearn.semi_supervised import SelfTrainingClassifier # for Semi-Supervised learning
from sklearn.metrics import classification_report # for model evaluation metrics
# Read in data
df = pd.read_csv('marketing_campaign.csv', encoding='utf-8', delimiter=';',usecols=['ID', 'Year_Birth', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome', 'MntWines', 'MntMeatProducts'])

#We will try to predict if our supermarket customer has dependents (children/teenagers) at home or not. Create a flag to denote whether the person has any dependants at home (either kids or teens)
df['Dependents_Flag']=df.apply(lambda x: 1 if x['Kidhome']+x['Teenhome']>0 else 0, axis=1)

#Test data will be used to evaluate model performance, while labeled and unlabeled data will be used to train our models.
df_train, df_test = train_test_split(df, test_size=0.25, random_state=0)
print('Size of train dataframe: ', df_train.shape[0])
print('Size of test dataframe: ', df_test.shape[0])

# Create a flag for label masking
df_train['Random_Mask'] = True
df_train.loc[df_train.sample(frac=0.05, random_state=0).index, 'Random_Mask'] = False

# Create a new target colum with labels. The 1's and 0's are original labels and -1 represents unlabeled (masked) data
df_train['Dependents_Target']=df_train.apply(lambda x: x['Dependents_Flag'] if x['Random_Mask']==False else -1, axis=1)

# Show target value distribution
print('Target Value Distribution:')
print(df_train['Dependents_Target'].value_counts())

#Basic Data Prep
# Select only records with known labels
df_train_labeled=df_train[df_train['Dependents_Target']!=-1]
# Select data for modeling
X_baseline=df_train_labeled[['MntMeatProducts', 'MntWines']]
y_baseline=df_train_labeled['Dependents_Target'].values
# Put test data into an array
X_test=df_test[['MntMeatProducts', 'MntWines']]
y_test=df_test['Dependents_Flag'].values

#Semi-Supervised model
#Data Prep (1) - Select data for modeling - we are including masked (-1) labels this time
X_train=df_train[['MntMeatProducts', 'MntWines']]
y_train=df_train['Dependents_Target'].values

#Model Fitting (2) - Specify SVC model parameters
model_svc = SVC(kernel='rbf',
                probability=True, # Need to enable to be able to use predict_proba
                C=1.0, # default = 1.0
                gamma='scale', # default = 'scale',
                random_state=0)
# Specify Self-Training model parameters
self_training_model = SelfTrainingClassifier(base_estimator=model_svc, # An estimator object implementing fit and predict_proba.
                                             threshold=0.7, # default=0.75, The decision threshold for use with criterion='threshold'. Should be in [0, 1).
                                             criterion='threshold', # {‘threshold’, ‘k_best’}, default=’threshold’,
                                             # The selection criterion used to select which labels to add to the training set. If 'threshold',
                                             # pseudo-labels with prediction probabilities above threshold are added to the dataset. If 'k_best',
                                             # the k_best pseudo-labels with highest prediction probabilities are added to the dataset.
                                             #k_best=50, # default=10, The amount of samples to add in each iteration. Only used when criterion='k_best'.
                                             max_iter=100, # default=10, Maximum number of iterations allowed. Should be greater than or equal to 0.
                                             # If it is None, the classifier will continue to predict labels until no new pseudo-labels are added, or all unlabeled samples have been labeled.
                                             verbose=True) # default=False, Verbosity prints some information after each iteration
# Fit the model
clf_ST = self_training_model.fit(X_train, y_train)
#Model Evaluation (3)
print('')
print('-------- Self Training Model - Summary --------')
print('Base Estimator: ', clf_ST.base_estimator_)
print('Classes: ', clf_ST.classes_)
print('Transduction Labels: ', clf_ST.transduction_)
print('Iteration When Sample Was Labeled: ', clf_ST.labeled_iter_)
print('Number of Features: ', clf_ST.n_features_in_)
print('Feature Names: ', clf_ST.feature_names_in_)
print('Number of Iterations: ', clf_ST.n_iter_)
print('Termination Condition: ', clf_ST.termination_condition_)
print('')
print('------- Self Training Model - Evaluation on Test Data -----')
accuracy_score_ST = clf_ST.score(X_test, y_test)
print('Accuracy Score %3f:' % (accuracy_score_ST*100))
# Look at classification report to evaluate the model
print(classification_report(y_test, clf_ST.predict(X_test)))




