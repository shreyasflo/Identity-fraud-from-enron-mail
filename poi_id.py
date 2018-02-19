# UDACITY DATA ANALYST NANODEGREE - Enron Fraud Detection
# 13-Feb-2018
# Shreyas Ramnath 

#!/usr/bin/python
import matplotlib.pyplot as plt
import sys
import pickle
from sklearn import preprocessing
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.grid_search import GridSearchCV
sys.path.append("../tools/")

from feature_format import featureFormat
from feature_format import targetFeatureSplit

### features_list is a list of strings, each of which is a feature name
### first feature must be "poi", as this will be singled out as the label

features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email", 'shared_receipt_with_poi']

### load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### look at data
#print len(data_dict.keys())
#print data_dict['BUY RICHARD B']
#print data_dict.values()


### remove any outliers before proceeding further
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)

### remove NAN's from dataset
outliers = []
for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    outliers.append((key, int(val)))

outliers_final = (sorted(outliers,key=lambda x:x[1],reverse=True)[:4])
### uncomment for printing top 4 salaries
### print outliers_final


### plot features
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
#plt.show()

### create new features
### new features are: fraction_to_poi_email,fraction_from_poi_email

def dict_to_list(key,normalizer):
    new_list=[]

    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][normalizer]=="NaN":
            new_list.append(0.)
        elif data_dict[i][key]>=0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return new_list

### create two lists of new features
fraction_from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")
fraction_to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")

### insert new features into data_dict
count=0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"]=fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"]=fraction_to_poi_email[count]
    count +=1

### store to my_dataset for easy export below
my_dataset = data_dict

### these two lines extract the features specified in features_list
### and extract them from data_dict, returning a numpy array
data = featureFormat(my_dataset, features_list)

### plot new features
for point in data:
    from_poi = point[1]
    to_poi = point[2]
    plt.scatter( from_poi, to_poi )
    if point[0] == 1:
        plt.scatter(from_poi, to_poi, color="r", marker="*")
plt.xlabel("fraction of emails this person gets from poi")
#plt.show()
### if you are creating new features, could also do that here
### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)

### machine learning goes here!
### please name your classifier clf for easy export below

### deploying feature selection based on importance scores
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.4, random_state=42)
features_list = ["poi", "salary", "bonus", "fraction_from_poi_email", "fraction_to_poi_email",
                 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']
ctr=DecisionTreeClassifier()
ctr.fit(features_train, labels_train)
pred = ctr.predict(features_test)
#print "accuracy ", accuracy_score(pred,labels_test)
importances = ctr.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
## Algorithms 
## Gaussian Naive Bayes

gnb = GaussianNB()
gnb.fit(features_train, labels_train)
pred = gnb.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
print '\n'
print "Naive Bayes :"
print "accuracy ", accuracy
print 'precision = ', precision_score(labels_test,pred)
print 'recall = ', recall_score(labels_test,pred)
print '\n'

## DecisionTreeClassifier Algorithm

print "DecisionTreeClassifier :"
dt = DecisionTreeClassifier()
dt.fit(features_train, labels_train)
pred = dt.predict(features_test)
print "accuracy ", accuracy_score(pred,labels_test)
print 'precision = ', precision_score(labels_test,pred)
print 'recall = ', recall_score(labels_test,pred)	

## Parameter tuning of the DecisionTreeClassifier ( Hyper Parameters)
dtc= DecisionTreeClassifier(random_state=0)
#Hyper Parameters Set
param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
			 'min_samples_split': [2,3,4,5,6,7,8,9,10], 
             'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10]
			 ,'random_state':[123]}
#Making models with hyper parameters sets
gscv = GridSearchCV(dtc, param_grid)
gscv.fit(features_train, labels_train)
#The best hyper parameters set
#print("Best Hyper Parameters:\n",gscv.best_params_)
	
## 10- Fold cross validation to validate our process 
from sklearn.cross_validation import KFold
precisions = []
recalls = []
feature_list = ["poi","salary","bonus","frac_from_poi_email","frac_to_poi_email","deferral_payments","total_payments","loan_advances"]
clf = DecisionTreeClassifier(max_features = 'auto', min_samples_split = 9, random_state = 123, min_samples_leaf = 2)
k_fold = KFold(len(labels),10)
for train_indices,test_indices in k_fold:
    clf.fit(features_train,labels_train)
    pred = clf.predict(features_test)
    precision =  precision_score(labels_test,pred)
    recall =  recall_score(labels_test,pred)
    precisions.append(precision)
    recalls.append(recall)
precison = sum(precisions)/len(precisions)
recall = sum(recalls)/len(recalls)
accuracy = accuracy_score(pred,labels_test)
print "\n","After tuning using GridSearchCV"
print "accuracy = ",accuracy
print "precision = ",precision 
print "recall = ",recall

## dump your classifier, dataset and features_list so
### that anyone can run/check your results

pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(data_dict, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )