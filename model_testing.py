# %%
# import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from statsmodels.tsa.api import VAR
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import math
# from sklearn.metrics import mean_squared_error
# from auto_ts import auto_timeseries

# grid search holt winter's exponential smoothing
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from numpy import array

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
		names += list(df.columns + ('_(t-%d)' % (i)))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += list(df.columns + ('_(t)'))
		else:
			names += list(df.columns + ('_(t+%d)' % (i)))
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def MinMax_Scaler(df):
  scalers = {}
  for idx,col in enumerate(df.columns):
    scaler = MinMaxScaler()
    ss = scaler.fit_transform(df[col].values.reshape(-1,1))
    ss = np.reshape(ss, len(ss))
    scalers[col] = scaler
    df[col] = ss
  return df, scalers

categorized_data, items_data, subcategory_data, category_data = preprocess(data_preprocessed)


# %%

# one-step Holt Winter’s Exponential Smoothing forecast
def exp_smoothing_forecast(history, config):
	t,d,s,p,b,r = config
	# define model
	history = array(history)
	model = ExponentialSmoothing(history, trend=t, damped=d, seasonal=s, seasonal_periods=p)
	# fit model
	model_fit = model.fit(optimized=True, use_boxcox=b, remove_bias=r)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = exp_smoothing_forecast(history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	return error

# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
	result = None
	# convert config to a key
	key = str(cfg)
	# show all warnings and fail on exception if debugging
	if debug:
		result = walk_forward_validation(data, n_test, cfg)
	else:
		# one failure during model validation suggests an unstable config
		try:
			# never show warnings when grid searching, too noisy
			with catch_warnings():
				filterwarnings("ignore")
				result = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
	# check for an interesting result
	if result is not None:
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
		# execute configs in parallel
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results
	scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

# create a set of exponential smoothing configs to try
def exp_smoothing_configs(seasonal=[None]):
	models = list()
	# define config lists
	t_params = ['add', 'mul', None]
	d_params = [True, False]
	s_params = ['add', 'mul', None]
	p_params = seasonal
	b_params = [True, False]
	r_params = [True, False]
	# create config instances
	for t in t_params:
		for d in d_params:
			for s in s_params:
				for p in p_params:
					for b in b_params:
						for r in r_params:
							cfg = [t,d,s,p,b,r]
							models.append(cfg)
	return models
# %%
# define dataset
data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
print(data)
# data split
n_test = 4
# model configs
cfg_list = exp_smoothing_configs()
# grid search
scores = grid_search(data, cfg_list, n_test)
print('done')
# list top 3 configs
for cfg, error in scores[:3]:
    print(cfg, error)
# %%
df = categorized_data['Fruits']['Apple']['galaapple'].resample('M').mean()
df_forecast = pd.DataFrame(df, index=df.index, columns = ['galaapple'])

df_forecast['HWES_additive'] = ExponentialSmoothing(df,trend='add',seasonal='add',seasonal_periods=12).fit().fittedvalues
df_forecast['HWES_multiplicative'] = ExponentialSmoothing(df,trend='mul',seasonal='mul',seasonal_periods=12).fit().fittedvalues

plt.plot(df_forecast)

# %%
