from sklearn.model_selection import RepeatedKFold, LeaveOneOut, LeavePOut, StratifiedKFold, GroupKFold, \
    StratifiedGroupKFold, LeaveOneGroupOut, LeavePGroupsOut, GroupShuffleSplit, TimeSeriesSplit, \
    StratifiedShuffleSplit
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold

# prepare of data
X, y = datasets.load_wine(return_X_y=True)  # load the wine data set to fit a linear support vector machine on it:
groups = y  # group parameter
scoring = ['precision_macro', 'recall_macro']  # paramteres for cross_validate
random_state = 200  # default parameter

print('Cross validation iterations')  ##Cross validation iterators
print('----------------------')

####### MODEL SVC
clf = svm.SVC(kernel='linear', C=1, random_state=0)
scores_val = cross_val_score(clf, X, y, cv=5)
scores_cross = cross_validate(clf, X, y, cv=5, scoring=scoring, return_train_score=True)
sorted(scores_cross.keys())
scores_cross1 = scores_cross['test_precision_macro']
scores_cross2 = scores_cross['test_recall_macro']
print('====== MODEL SVC ======')
print("1. Number of SVC,  scores used in Average: %s" % (len(scores_val)))
print( f"2. Cross_val_scores -> accuracy: {scores_val.mean() * 100}, standard deviation: {scores_val.std()}")  # The value of the mean and the 95% confidence interval of the estimate of the results
print(f"3. Cross_validate   -> precision: {scores_cross1}, recall: {scores_cross2}")
print("-------------------------------------------------------------------------------------------------------------------------------------------")

# defined functions to calculate cross_validate and cross_val_score for single and group scores
def printMetrics(name, cv):  # single
    print("\n ====== %s ======" % name)
    print(cv)
    scores_val = cross_val_score(clf, X=X, y=y, cv=cv, )
    cross = cross_validate(clf, X=X, y=y, cv=cv, scoring=scoring, return_train_score=True, return_estimator=True)
    sorted(cross.keys())
    scores1 = cross['test_precision_macro']
    scores2 = cross['test_recall_macro']
    print("1. Number of %s,  scores used in Average: %s" % (name, len(scores_val)))
    print( f"2. Cross_val_scores -> accuracy: {scores_val.mean() * 100}, standard deviation: {scores_val.std()}")  # The value of the mean and the 95% confidence interval of the estimate of the results
    print(f"3. Cross_validate   -> precision: {scores1}, recall: {scores2}")

def printMetricsGroup(name, cv):  # group
    print("\n ====== %s ======" % name)
    print(cv)
    scores_val = cross_val_score(clf, X=X, y=y, cv=cv, groups=groups)
    cross = cross_validate(clf, X=X, y=y, cv=cv, scoring=scoring, return_train_score=True, return_estimator=True)
    sorted(cross.keys())
    scores1 = cross['test_precision_macro']
    scores2 = cross['test_recall_macro']
    print("1. Number of %s,  scores used in Average: %s" % (name, len(scores_val)))
    print(f"2. Cross_val_scores -> accuracy: {scores_val.mean() * 100}, standard deviation: {scores_val.std()}")  # The value of the mean and the 95% confidence interval of the estimate of the results
    print(f"3. Cross_validate   -> precision: {scores1}, recall: {scores2}")

#### Cross-validation iterators for i.i.d. data ####
print("Cross-validation iterators for i.i.d. data")
# KFold
kf = KFold(n_splits=2)
printMetrics("KFold", kf)

# Repeated KFold
rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
printMetrics("Repeated KFold", rkf)

# LOO
loo = LeaveOneOut()
printMetrics("Leave One Out", loo)

# Leave P Out (LPO)
lpo = LeavePOut(p=2)
printMetrics("Leave P Out", lpo)

####Cross-validation iterators with stratification based on class labels ####
print("Cross-validation iterators with stratification based on class labels ")
# Stratified k-fold
skf = StratifiedKFold(n_splits=3)
printMetrics("Stratified kFold", skf)

# Stratified Shuffle Spilt
sss = StratifiedShuffleSplit(n_splits=3)
printMetrics("Stratified Shuffle Spilt", sss)

# Random permutations cross-validation a.k.a. Shuffle & Split
ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
printMetrics("Shuffle & Split", ss)

#### Cross-validation iterators for grouped data ####
print("Cross-validation iterators for grouped data")
# Group kFold
gkf = GroupKFold(n_splits=2).get_n_splits(X, y, groups)
printMetricsGroup("Group kFold", gkf)

# StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=3).get_n_splits(X, y, groups)
printMetricsGroup("StratifiedGroupKFold", sgkf)

# Leave One Group Out
logo = LeaveOneGroupOut().get_n_splits(X, y, groups)
printMetricsGroup("Leave One Group Out", sgkf)

# Leave P Groups Out
lpgo = LeavePGroupsOut(n_groups=2).get_n_splits(X, y, groups)
printMetricsGroup("Leave P Groups Out", lpgo)

# Group Shuffle Split
gss = GroupShuffleSplit(n_splits=2, test_size=0.5, random_state=0).get_n_splits(X, y, groups)
printMetricsGroup("Group Shuffle Split", gss)

#### Cross validation of time series data ####
print("Cross validation of time series data")
# Time Series Split
tscv = TimeSeriesSplit(n_splits=3)
printMetrics("Time Series Split", tscv)
