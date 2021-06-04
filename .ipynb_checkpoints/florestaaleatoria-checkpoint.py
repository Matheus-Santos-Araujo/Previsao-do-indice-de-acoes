spark = SparkSession.builder.appName('classification').getOrCreate()

from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
plt.style.use('ggplot')

dfp = pd.read_csv('previsaodeacoes.csv')

plt.figure(figsize=(12,8))
sns.heatmap(dfp.describe()[1:].transpose(),
            annot=True,linecolor="w",
            linewidth=2,cmap=sns.color_palette("Set2"))
plt.title("Sumario")
plt.show()

cor_mat= dfp[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)

explode = (0.1,0)  
fig1, ax1 = plt.subplots(figsize=(12,7))
ax1.pie(dfp['LABEL'].value_counts(), explode=explode,labels=['Down','Up'], autopct='%1.1f%%',
        shadow=True)

ax1.axis('igual')  
plt.tight_layout()
plt.legend()
plt.show()

plt.figure(figsize=(10,9))
sns.scatterplot(x='InterestRate',y='ExchangeRate',data=dfp,palette='Set1', hue = 'LABEL');

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
model = RandomForestClassifier(labelCol='label')
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

real = np.array(test_df.select("label").collect())
predito = np.array(test_predictions.select("label").collect())

cm = confusion_matrix(real, predito)
class_names=[0,1] 
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="RdGy" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Matriz de confusão', y=1.1)
plt.ylabel('Label real')
plt.xlabel('Label predita')

