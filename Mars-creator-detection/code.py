# --------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd


# Load dataset using pandas read_csv api in variable df
df = pd.read_csv(path)
# Dispay first 5 columns
df.head(5)

## As we can observe we have total 1088 independent variables and attr1089 is the dependent variable lets split them
X = df.drop(columns='attr1089')
y = df['attr1089']

## Lets split the data for Train test
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 4)

## Intializing minmax Scaler
scaler = MinMaxScaler()

# Fit the scaler on X_train
scaler.fit(X_train)

# Fit and transform X_train and X_test
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# --------------
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


lr = LogisticRegression(max_iter=1500)
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

roc_score = roc_auc_score(y_test,y_pred)
print(roc_score)


# --------------
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=4)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
roc_score = roc_auc_score(y_test,y_pred)
print(roc_score)


# --------------
from sklearn.ensemble import RandomForestClassifier


rfc = RandomForestClassifier(random_state=4)
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
roc_score = roc_auc_score(y_test,y_pred)
print(roc_score)





# --------------
# Import Bagging Classifier
from sklearn.ensemble import BaggingClassifier


# Code starts here
bagging_clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100,max_samples=100,random_state=0)
bagging_clf.fit(X_train,y_train)
score_bagging = bagging_clf.score(X_test,y_test)
print(score_bagging)

# Code ends here


# --------------
# Import libraries
from sklearn.ensemble import VotingClassifier

# Various models
clf_1 = LogisticRegression()
clf_2 = DecisionTreeClassifier(random_state=4)
clf_3 = RandomForestClassifier(random_state=4)

model_list = [('lr',clf_1),('DT',clf_2),('RF',clf_3)]


# Code starts here
voting_clf_hard = VotingClassifier(estimators=model_list,voting='hard')
voting_clf_hard.fit(X_train,y_train)
hard_voting_score = voting_clf_hard.score(X_test,y_test)
print(hard_voting_score)

# Code ends here


