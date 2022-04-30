import importlib
import header
importlib.reload(header) # For reloading after making changes
from header import *

class XGBoost:
	def __init__(self, X_train, X_test, y_train, y_test):
		self.model = False
		self.evaluate_acc = False
		self.evaluate_cm = False
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test
		self.xgb_model()
		self.recall = False
		self.precision = False
		self.importanceFeatures(self.model)
		# self.tree(self.model)
	def xgb_model(self):
		print('Running  XGBoost model')
		# XGBClassifier initialisation
		self.model = XGBClassifier(objective='reg:squarederror', learning_rate=0.1, max_depth=5, colsample_bytree=0.3)
		self.model.fit(self.X_train, self.y_train)
		# evaluation the model
		self.evaluate_acc = self.model_evaluate(self.model)
		self.evaluate_cm = self.model_evaluate(self.model)
		self.recall = self.model_evaluate(self.model)
		self.precision = self.model_evaluate(self.model)
	def model_evaluate(self, model):
		# make predictions using test set
		y_pred = model.predict(self.X_test)
		# Calculating accuracy
		accuracy = accuracy_score(self.y_test, y_pred)
		# Building confusion matrix
		cm = confusion_matrix(self.y_test, y_pred)
		precision= precision_score(self.y_test, y_pred)
		recall = recall_score(self.y_test, y_pred)
		return (accuracy, cm, precision,recall)
	def importanceFeatures(self, model):
		# plt.rcParams['figure.figsize'] = [16, 15]
		# plt.rcParams['figure.dpi'] = 100
		plt.rc('axes', axisbelow=True)
		xgb.plot_importance(model)
		plt.show()
	def tree(self, model):
		plt.rcParams['figure.figsize'] = [7, 5]
		plt.rcParams['figure.dpi'] = 150
		xgb.plot_tree(model)
		plt.show()



if __name__ == '__main__':
	X_train, X_test, y_train, y_test = dataset_loader.DatasetLoader('./assets/ISCX_Botnet.csv')
	# run XGBoost, print Acc score, CM and plots
	xgb = XGBoost(X_train, X_test, y_train, y_test)
	print('XGBoost accuracy score: {}'.format(xgb.evaluate_acc[0]*100.00))
	print(xgb.evaluate_cm[1])
	print('Precesion: ',xgb.evaluate_cm[2])
	print('Recall: ',xgb.evaluate_cm[3])
