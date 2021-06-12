import numpy as np 
import pandas as pd 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
from sklearn.metrics import confusion_matrix
from sklearn import metrics

df = pd.read_csv('previsaodeacoes.csv')
df.head(10)

plt.figure(figsize=(12,8))
sns.heatmap(df.describe()[1:].transpose(),
            annot=True,linecolor="w",
            linewidth=2,cmap=sns.color_palette("Set2"))
plt.title("Sumário")
plt.show()
plt.savefig('sumario.png')

cor_mat= df[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)
plt.savefig('matrizcorrelacao.png')

corr=df.corr()
corr.sort_values(by=["LABEL"],ascending=False).iloc[0].sort_values(ascending=False)

explode = (0.1,0)  
fig1, ax1 = plt.subplots(figsize=(12,7))
ax1.pie(df['LABEL'].value_counts(), explode=explode,labels=['Down','Up'], autopct='%1.1f%%',
        shadow=True)
 Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()
plt.savefig('divisao.png')

plt.figure(figsize=(10,9))
sns.scatterplot(x='InterestRate',y='ExchangeRate',data=df,palette='Set1', hue = 'LABEL');
plt.savefig('amostras.png')