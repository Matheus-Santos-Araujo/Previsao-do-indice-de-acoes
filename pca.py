import findspark
from pyspark.sql import SparkSession

findspark.init()

spark = SparkSession.builder \
        .master("local[*]") \
        .appName("pca") \
        .getOrCreate()

from pyspark.ml.feature import PCA as PCAml
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import NaiveBayes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pyspark.ml.feature import Normalizer
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
df = transformColumnsToNumeric (df, "Close")
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
#normalizer = Normalizer(inputCol=inputCols, outputCol="normfeatures", p=1.0)
#df = normalizer.transform(df)
df_transformed = df.select(['features','label'])

#normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)
#df_transformed = normalizer.transform(df_transformed)

kcomp = 2

pca = PCAml(k=kcomp, inputCol="features", outputCol="pca")
model = pca.fit(df_transformed)
transformed = model.transform(df_transformed)
transformed.printSchema()

pca_var= np.round(100.00*model.explainedVariance.toArray(),kcomp)
exvar_pca = sum(model.explainedVariance)

print("Relevance of each component")
print(pca_var)

result = model.transform(df_transformed).select("pca")
result.show(truncate=False)

colname = [] 
for i in range(1, kcomp+1):
    colname.append("PC"+str(i))

name = colname
values = pca_var 

fig, ax = plt.subplots(figsize=(9, 3), sharey=True)
ax.bar(name, values)
fig.suptitle('Relevancia de cada componente')
plt.savefig("pca.png")

# train_df, test_df = transformed.randomSplit([0.75,0.25])
# model = DecisionTreeClassifier(labelCol='label', featuresCol='normFeatures')
# trained_model = model.fit(train_df)
# test_predictions = trained_model.transform(test_df)