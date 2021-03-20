import pandas as p
from pandas import DataFrame as df
import matplotlib.pyplot as pl
from sklearn.linear_model import LinearRegression

data=p.read_csv('cost_revenue_clean.csv')
x=df(data,columns=['production_budget'])
y=df(data,columns=['worldwide_gross'])

pl.figure(figsize=(20,10))
pl.scatter(x,y,alpha=0.5)
pl.title('movie data')
pl.xlabel('production cost')
pl.ylabel('worldwide gross')
regression=LinearRegression()
regression.fit(x,y)
pl.plot(x,regression.predict(x),color='red',linewidth=4)
pl.show()
