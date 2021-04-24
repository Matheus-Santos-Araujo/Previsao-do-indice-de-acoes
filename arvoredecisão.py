spark = SparkSession.builder.appName('classification').getOrCreate()

from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier


df = spark.read.csv('previsaoacoes.csv', inferSchema=True, header=True)

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

train_df, test_df = df_transformed.randomSplit([0.75,0.25])
model = DecisionTreeClassifier(labelCol='label')
trained_model = model.fit(train_df)

test_predictions = trained_model.transform(test_df)

test_df_count_1 = test_df.filter(test_df['label'] == 1).count()
test_df_count_0 = test_df.filter(test_df['label'] == 0).count()
test_df_count_1, test_df_count_0

cp = test_predictions.filter(
test_predictions['label'] == 1).filter(
test_predictions['prediction'] == 1).select(
['label','prediction','probability'])
print("Predições corretas: ", cp.count())
accuracy = (cp.count()) /  test_df_count_1
print(f"Acurácia: {accuracy}\n")

fp = train_predictions.filter(
test_predictions['label'] == 0).filter(
test_predictions['prediction'] == 1).select(
print("Falsos positivos: ", fp.count())

fn = train_predictions.filter(
test_predictions['label'] == 1).filter(
test_predictions['prediction'] == 0).select(
['label','prediction','probability'])
print("Falsos negativos: ", fn.count())
