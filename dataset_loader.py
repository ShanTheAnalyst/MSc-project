import importlib
import header
importlib.reload(header) # For reloading after making changes
from header import *


class DatasetLoader:
	def __init__(self):
		return None
	def __new__(self, path):
		# laod scv from the path
		dataset = pd.read_csv(path)
		print(dataset['Label'].value_counts())
		# selection of all columns except label class with label = 0
		X = dataset.iloc[:, :-1].values
		# selection of only label class with label = 1
		y = dataset.iloc[:, -1].values
		# spliting of data into X_train, X_test, y_train and y_test
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
		return (X_train, X_test, y_train, y_test)