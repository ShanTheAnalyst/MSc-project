import header
from header import *

class KNN:
	def __init__(self, X_train, X_test, y_train, y_test):
		self.model = False
		self.evaluate_acc = False
		self.evaluate_cm = False
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test
		self.classifier()
	def classifier(self):
		print('Running model')
		# XGBClassifier initialisation
		self.model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
		# Training of model
		self.model.fit(self.X_train, self.y_train)
		# Evaluation the model
		self.evaluate_acc = self.model_evaluate(self.model)
		self.evaluate_cm = self.model_evaluate(self.model)
	def model_evaluate(self, model):
		# Make predictions using test set
		y_pred = model.predict(self.X_test)
		# Calculating accuracy
		accuracy = accuracy_score(self.y_test, y_pred)
		# Building confusion matrix
		cm = confusion_matrix(y_test, y_pred)
		return (accuracy, cm)

if __name__ == '__main__':
	X_train, X_test, y_train, y_test = dataset_loader.DatasetLoader('./assets/ISCX_Botnet(Prep).csv')
	# Run model and print accuracy score and cm
	knn = KNN(X_train, X_test, y_train, y_test)
	print('Model accuracy score: {}'.format(knn.evaluate_acc[0]*100.00))
	print(knn.evaluate_cm[1])
