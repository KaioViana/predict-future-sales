# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from  datetime import datetime, date
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler
from math import ceil
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop


%matplotlib inline

train = pd.read_csv(r'Data/sales_train.csv')
test = pd.read_csv(r'Data/test.csv')
submission = pd.read_csv(r'Data/sample_submission.csv')
items = pd.read_csv(r'Data/items.csv')
item_cats = pd.read_csv(r'Data/item_categories.csv')
shops = pd.read_csv(r'Data/shops.csv')

# Primeiro verificaremos se todos os shops e itens do arquivo de test esta no de treino
test_shops = test.shop_id.unique()
train = train[train.shop_id.isin(test_shops)]
test_items = test.item_id.unique()
train = train[train.item_id.isin(test_items)]

MAX_BLOCK_NUM = train.date_block_num.max()
MAX_ITEM = len(test_items)
MAX_CAT = len(item_cats)
MAX_YEAR = 3
MAX_MONTH = 4
MAX_SHOP = len(test_shops)

# agrupando pelo shop_id e date_block_num a soma dos itens vendidos
grouped = pd.DataFrame(train.groupby(['shop_id', 'date_block_num'])['item_cnt_day'].sum().reset_index())
fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(16,16))
num_graph = 10
id_per_graph = ceil(grouped.shop_id.max() / num_graph)
count = 0
for i in range(5):
    for j in range(2):
        sns.pointplot(
                x='date_block_num', y='item_cnt_day', hue='shop_id',
                data=grouped[np.logical_and(
                        count*id_per_graph <= grouped['shop_id'],
                        grouped['shop_id'] < (count+1)*id_per_graph)], ax=axes[i][j])
        count += 1

# analizando o gráfico gerado, pode-se ver que há um maior pico no fim do ano.
# seria interessante então adicionar o mês e o ano para que a rede possa captar esse padrão.
# seria interessante também ver como está indo a venda de cada item.
# no entanto, dado o número de itens, seria mais benéfico se analisarmos o desempenho de cada categoria de item

# add categorias / mês
train = train.set_index('item_id').join(items.set_index('item_id')).drop('item_name', axis=1).reset_index()
train['month'] = train.date.apply(lambda x: datetime.strptime(x, '%d.%m.%Y').strftime('%m'))
train['year'] = train.date.apply(lambda x: datetime.strptime(x,'%d.%m.%Y').strftime('%Y'))

fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(16,20))
num_graph = 10
id_per_graph = ceil(train.item_category_id.max() / num_graph)
count = 0
for i in range(5):
    for j in range(2):
        sns.pointplot(x='month', y='item_cnt_day', hue='item_category_id',
                      data=train[np.logical_and(
                              count*id_per_graph <= train['item_category_id'],
                              train['item_category_id'] < (count+1)*id_per_graph)], ax=axes[i][j])
        count += 1


# add categorias / ano
fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(16,20))
num_graph = 10
id_per_graph = ceil(train.item_category_id.max() / num_graph)
count = 0
for i in range(5):
    for j in range(2):
        sns.pointplot(x='date_block_num', y='item_cnt_day', hue='item_category_id',
                      data=train[np.logical_and(count*id_per_graph <= train['item_category_id'], train['item_category_id'] < (count+1)*id_per_graph)],
                      ax=axes[i][j])
        count += 1


train = train.drop('date', axis=1)
train = train.drop('item_category_id', axis=1)
train = train.groupby(['shop_id', 'item_id', 'date_block_num', 'month', 'year']).sum()
train = train.sort_index()

# Treinamento
# No método de de aprendizado baseado em gradiente, é comum normalizar a variável
# numérica para acelerar o treinamento
scaler = StandardScaler()
cnt_scaler = StandardScaler()

scaler.fit(train.item_price.as_matrix().reshape(-1, 1))
cnt_scaler.fit(train.item_cnt_day.as_matrix().reshape(-1, 1))

train.item_price = scaler.transform(train.item_price.as_matrix().reshape(-1, 1))
train.item_cnt_day = cnt_scaler.transform(train.item_cnt_day.as_matrix().reshape(-1, 1))

# É improvável que os dados de venda de janeiro de 2013 ou a qualquer momento por perto
# tenham algum efeito com a venda de novembro de 2015.

# Em vez disso, aprenderíamos a sequência de julho, agosto, setembro, outubro, novembro em 2013 e 14.

