import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor



bost_df = pd.read_csv('C:\\Users\\aadya.j\\Desktop\\Aadya\\23.12.19\\Boston.csv')

x=bost_df.drop('medv',axis=1)
y=bost_df['medv']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, )

regressor = DecisionTreeRegressor(criterion='mse',random_state=100,max_depth=4,min_samples_leaf=1)
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})

print(df)


