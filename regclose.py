import findspark
from pyspark.sql import SparkSession

findspark.init()

spark = SparkSession.builder \
        .master("local[*]") \
        .appName("regressaologistica") \
        .getOrCreate()

from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.regression import DecisionTreeRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
plt.style.use('ggplot')

df = spark.read.csv('previsaodeacoes.csv', inferSchema=True, header=True)

class_indexer = StringIndexer(inputCol="Close", outputCol="close")

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
df_transformed = df.select(['features','close'])

normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)
df_transformed = normalizer.transform(df_transformed)

train_df, test_df = df_transformed.randomSplit([0.75,0.25])
model = RandomForestRegressor(labelCol='close', featuresCol='features')

# Fit the model
trained_model = model.fit(train_df)

test_predictions = trained_model.transform(test_df)

real = np.array(test_df.select("close").collect())
predito = np.array(test_predictions.select("prediction").collect())

RMSE = mean_squared_error(real, predito, squared=False)
print(RMSE)
        
MAE = mean_absolute_error(real, predito)
print(MAE)

