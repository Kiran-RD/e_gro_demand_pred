# %%
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.plt as plt
import math
from sklearn.metrics import mean_squared_error
import seaborn as sns
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense,LSTM,GRU

# %%
data = pd.read_csv('Customer_Data_unsorted1.csv')
data_preprocessed = data[['order_date','basket']]
data_preprocessed.order_date = pd.to_datetime(data_preprocessed.order_date)

# Preprocess data
def preprocess(data_preprocessed):


    def get_quant(quant):
        if 'kg' in quant.lower() or 'pc' in quant.lower():
            num = float(quant[:-2])
        elif 'l' in quant.lower():
            num = float(quant[:-1])
        else:
            num = float(quant[:-1])/1000
        return num


    items_dict = {} # Dict to get row wise mapping of Quantity for every item
    for col in ['basket']:
    # for col in ['order_fruits','order_vegetables','order_milk','order_rice']:
        print(col) # Debug Print
        tmp_df = data_preprocessed[col].str.split(',',expand=True) # Splitting on ',' to separate items. Expand splits items into columns
        for tmp_col in tmp_df.columns: # 
            print(tmp_col) # Debug Print
            tmp_series = tmp_df[tmp_col].str.split('-') # Splitting on '-' to separate Quantities
            for row in range(len(tmp_series)):
                element = tmp_series[row]
                # print(element) # Debug Print
                if isinstance(element,list):
                    #print(element)
                    if element[0].lower().replace(' ','') in items_dict:
                        if len(element) == 2:
                            items_dict[element[0].lower().replace(' ','')][row] = get_quant(element[1].strip()) 
                        else:
                            items_dict[element[0].lower().replace(' ','')][row] = np.nan
                    else:
                        if len(element) == 2:
                            items_dict[element[0].lower().replace(' ','')] = [np.nan]*data_preprocessed.shape[0]
                            items_dict[element[0].lower().replace(' ','')][row] = get_quant(element[1].strip())
                        else:
                            items_dict[element[0].lower().replace(' ','')] = [np.nan]*data_preprocessed.shape[0]
                            items_dict[element[0].lower().replace(' ','')][row] = np.nan

    # items_dict.pop('')
    items = pd.DataFrame(items_dict)
    items['order_date'] = data_preprocessed.order_date
    items.sort_values('order_date',inplace=True)
    items = items.groupby('order_date').sum()

    cat_lst = []
    for i in items.columns:
        # Fruits
        if 'sapota' in i:
            cat_lst.append(('Fruits','Sapota',i))
        elif 'custardapple' in i:
            cat_lst.append(('Fruits','Custardapple',i))
        elif 'apple' in i:
            cat_lst.append(('Fruits','Apple',i))
        elif 'pomogranate' in i or 'pomograntes' in i or 'pomegranetes' in i:
            cat_lst.append(('Fruits','Pomogranate',i))
        elif 'banana' in i:
            cat_lst.append(('Fruits','Banana',i))
        elif 'papaya' in i:
            cat_lst.append(('Fruits','Papaya',i))
        elif 'melon' in i:
            cat_lst.append(('Fruits','Melon',i))
        elif 'mango' in i:
            cat_lst.append(('Fruits','Mango',i))

        # Vegetables
        elif 'carrot' in i or 'gaajar' in i:
            cat_lst.append(('Veges','Carrot',i))
        elif 'onion' in i:
            cat_lst.append(('Veges','Onion',i))
        elif 'tomato' in i:
            cat_lst.append(('Veges','Tomato',i))
        elif 'chilli' in i:
            cat_lst.append(('Veges','Chilli',i))
        elif 'capsicum' in i:
            cat_lst.append(('Veges','Capsicum',i))
        elif 'brocoli' in i:
            cat_lst.append(('Veges','Brocoli',i))
        elif 'pepper' in i:
            cat_lst.append(('Veges','Pepper',i))

        # Milk
        elif 'milk' in i :
            cat_lst.append(('Milk','Milk',i))
        
        # Rice
        elif 'basmati' in i:
            cat_lst.append(('Rice','Basmati',i))
        elif 'brown' in i:
            cat_lst.append(('Rice','Brown',i))
        elif 'sonamasuri' in i:
            cat_lst.append(('Rice','Sonamasuri',i))
        elif 'organictattva' in i or 'daawat' in i:
            cat_lst.append(('Rice','Others',i))
        else:
            cat_lst.append((i,i,i))

    categorized_data = pd.DataFrame(items.values,index=items.index,columns=pd.MultiIndex.from_tuples(cat_lst,names=['category', 'subcategory','items']))
    categorized_data.sort_index(axis=1,level=['category','subcategory','items'],inplace=True)
    categorized_data.head()

    items_data = categorized_data.sum(axis=1,level='items')
    subcategory_data = categorized_data.sum(axis=1,level='subcategory')
    category_data = categorized_data.sum(axis=1,level='category')

    return categorized_data, items_data, subcategory_data, category_data

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += list(df.columns + ('(t-%d)' % (i)))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += list(df.columns + ('(t)'))
		else:
			names += list(df.columns + ('(t+%d)' % (i)))
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# %%
categorized_data, items_data, subcategory_data, category_data = preprocess(data_preprocessed)

# %%
df = series_to_supervised(categorized_data.Fruits.Apple,6,1)
df.head()
# %%
# split into train and test sets
values = df.iloc[:,:-2].values
n_train_days = 600
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
# %%
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
# %%
