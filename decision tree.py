import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.datasets import load_boston

boston = load_boston()
bost_df = pd.DataFrame(boston['data'], columns=boston.feature_names)
bost_df['MEDV']=boston.target


x = bost_df[[ 'RM','LSTAT', 'PTRATIO']]
y=boston.target

xv, xt, yv, yt = train_test_split(x, y, test_size = 0.2, random_state=5)
clf = DecisionTreeClassifier()
clf = clf.fit(xt,yt)
