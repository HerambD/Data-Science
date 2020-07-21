# --------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# path- variable storing file path
df = pd.read_csv(path)
df.head(5)

# Splitting data
X = df.drop('Price',axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 6)

# correlation between the features that are stored in 'X_train'
corr = X_train.corr()
corr
#Code starts here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Code starts here
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)


## R2 score
r2 = r2_score(y_test,y_pred)
print("Ther r2 Score for above model is == " + str(r2))


# --------------
from sklearn.linear_model import Lasso

# Code starts here
lasso = Lasso()
lasso.fit(X_train,y_train)
lasso_pred = lasso.predict(X_test)

# R2 score
r2_lasso = r2_score(y_test,y_pred)
print("The r2 Score for Lasso model is == " + str(r2_lasso))


# --------------
from sklearn.linear_model import Ridge

# Code starts here
# Code starts here
ridge = Ridge()
ridge.fit(X_train,y_train)
ridge_pred = ridge.predict(X_test)

# R2 score
r2_ridge = r2_score(y_test,y_pred)
print("The r2 Score for Lasso model is == " + str(r2_ridge))


# Code ends here


# --------------
from sklearn.model_selection import cross_val_score

#Code starts here
regressor = LinearRegression()
score = cross_val_score(regressor, X_train, y_train, cv=10)
mean_score = np.mean(score)
print("The mean score is == " + str(mean_score))


# --------------
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

#Code starts here
model = make_pipeline(PolynomialFeatures(2),LinearRegression())
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

# R2 score
r2_poly = r2_score(y_test,y_pred)
print("The R^2 score for using Polynomial Regression is = " + str(r2_poly))


