import findspark
from pyspark.sql import SparkSession

findspark.init()

spark = SparkSession.builder \
        .master("local[*]") \
        .appName("arvoredecisao") \
        .getOrCreate()

from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pyspark.ml.feature import Normalizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
plt.style.use('ggplot')

df = spark.read.csv('previsaodeacoes.csv', inferSchema=True, header=True)

class_indexer = StringIndexer(inputCol="LABEL", outputCol="label")

df = class_indexer.fit(df).transform(df)

def transformColumnsToNumeric(df, inputCol):
    
    inputCol_indexer = StringIndexer(inputCol = inputCol, outputCol = inputCol + "-index").fit(df)
    df = inputCol_indexer.transform(df)
    
    onehotencoder_vector = OneHotEncoder(inputCol = inputCol + "-index", outputCol = inputCol + "-vector")
    df = onehotencoder_vector.fit(df).transform(df)
    
    return df
    pass

df = transformColumnsToNumeric (df, "Open") 
df = transformColumnsToNumeric (df, "High") 
df = transformColumnsToNumeric (df, "Low") 
df = transformColumnsToNumeric (df,  "Close")
df = transformColumnsToNumeric (df, "Volume") 
df = transformColumnsToNumeric (df, "InterestRate") 
df = transformColumnsToNumeric (df, "ExchangeRate")
df = transformColumnsToNumeric (df, "VIX") 
df = transformColumnsToNumeric (df, "Gold") 
df = transformColumnsToNumeric (df, "Oil") 
df = transformColumnsToNumeric (df, "TEDSpread") 
df = transformColumnsToNumeric (df, "EFFR")  

inputCols=[
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

df_va = VectorAssembler(inputCols = inputCols, outputCol="features")
df = df_va.transform(df)
df_transformed = df.select(['features','label'])

normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)
df_transformed = normalizer.transform(df_transformed)

train_df, test_df = df_transformed.randomSplit([0.75,0.25])
model = DecisionTreeClassifier(labelCol='label', featuresCol='normFeatures')
trained_model = model.fit(train_df)

test_predictions = trained_model.transform(test_df)

test_df_count_1 = test_df.filter(test_df['label'] == 1).count()
test_df_count_0 = test_df.filter(test_df['label'] == 0).count()
test_df_count_1, test_df_count_0

fp = test_predictions.filter(
test_predictions['label'] == 0).filter(
test_predictions['prediction'] == 1).select(
['label','prediction','probability'])

print("Falsos positivos: ", fp.count())

fn = test_predictions.filter(
test_predictions['label'] == 1).filter(
test_predictions['prediction'] == 0).select(
['label','prediction','probability'])

print("Falsos negativos: ", fn.count())

predictionAndLabels = test_predictions.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Acurácia = " + str(evaluator.evaluate(predictionAndLabels)))

real = np.array(test_df.select("label").collect())
predito = np.array(test_predictions.select("prediction").collect())

cm = confusion_matrix(real, predito)
fig, ax = plt.subplots()
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="RdGy" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Matriz de confusão', y=1.1)
plt.ylabel('Label real')
plt.xlabel('Label predita')
plt.savefig('arvoredecisao.png')

