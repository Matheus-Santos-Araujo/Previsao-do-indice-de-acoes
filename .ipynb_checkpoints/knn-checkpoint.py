from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn import preprocessing
plt.style.use('ggplot')

dataset = pd.read_csv('previsaodeacoes.csv')

feature_columns = [
	'Open',
	'High', 
	'Low',	
	'Close',	
	'Volume',	
	'InterestRate',	
	'ExchangeRate',	
	'VIX',	
	'Gold',	
	'Oil',	
	'TEDSpread',	
	'EFFR']
X = dataset[feature_columns].values
y = dataset['LABEL'].values

# Split ramdomly the dataset into train and test group
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

classifier = KNeighborsClassifier(n_neighbors=2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred)*100
time_knn = time.time() - start_time_knn
print('KNN accuracy = ' + str(round(accuracy_knn, 1)) + ' %.')
print("K-Nearest Neighbors (KNN): accuracy = %3.1f %%" % accuracy_knn)

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="RdGy" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Matriz de confusão', y=1.1)
plt.ylabel('Label real')
plt.xlabel('Label predita')
plt.savefig('knn.png')