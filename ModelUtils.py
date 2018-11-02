import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

def read_dataset(path):
	return pd.read_csv(path)
	

def pre_processing(dataset):
	## feature engineering
	dataset = dataset[['Total','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed','Generation','Legendary']]
	## label encoding for legendary
	dataset['Legendary'] = LabelEncoder().fit_transform(dataset['Legendary'])
	return dataset
	

def split_dataset(dataset):
	X = dataset.iloc[:,[0,1,2,3,4,5,6,7]].values
	Y = dataset.iloc[:,[8]].values
	return X,Y
	

def train(algorithm_name,x_train,y_train):
	if algorithm_name == "lr":
		return LogisticRegression().fit(x_train,y_train)
	elif algorithm_name == "svm":
		return svm.SVC().fit(x_train,y_train)
	elif algorithm_name == "dt":
		return DecisionTreeClassifier().fit(x_train,y_train)
	elif algorithm_name == "knn":
		return KNeighborsClassifier(n_neighbors=3).fit(x_train,y_train)
	elif algorithm_name == "nb":
		return GaussianNB().fit(x_train,y_train)
	else:
		return "Error in Choosing training algorithm"
		

def evaluation(model,x_test,y_test):
	y_pred = model.predict(x_test)
	accuracy = accuracy_score(y_test,y_pred)
	cm = confusion_matrix(y_test,y_pred)
	report = classification_report(y_test,y_pred)
	return accuracy,cm,report
