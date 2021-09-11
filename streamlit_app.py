# %%
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Use the non-interactive Agg backend, which is recommended as a
# thread-safe backend.
# See https://matplotlib.org/3.3.2/faq/howto_faq.html#working-with-threads.
import matplotlib as mpl
mpl.use("agg")
# import mysql.connector # For data base connection

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

import math
from sklearn.metrics import mean_squared_error


##############################################################################
# Workaround for the limited multi-threading support in matplotlib.
# Per the docs, we will avoid using `matplotlib.pyplot` for figures:
# https://matplotlib.org/3.3.2/faq/howto_faq.html#how-to-use-matplotlib-in-a-web-application-server.
# Moreover, we will guard all operations on the figure instances by the
# class-level lock in the Agg backend.
##############################################################################
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock


# %%
# -- Set page config
apptitle = 'Demand Prediction'
st.set_page_config(page_title=apptitle, page_icon=":eyeglasses:")

# st.markdown("""
#  * Use the menu at left to select data and set plot parameters
#  * Your plots will appear below
# """)

st.title('Demand Prediction')

# %%
@st.cache
def get_data_from_database():
    data = pd.read_csv('Customer_Data_unsorted1.csv')
    data_preprocessed = data[['order_date','basket']]
    data_preprocessed.order_date = pd.to_datetime(data_preprocessed.order_date)
    return data, data_preprocessed

    # connection = mysql.connector.connect(host='localhost',
    #                                     database='e_gro',
    #                                     user='root',
    #                                     password='roottoor')
    # # connection.get_server_info()
    # cursor = connection.cursor()
    # cursor.execute("select * from sales;")
    # records = cursor.fetchall()
    # # for i in cursor.fetchall():
    # #     print(i)
    # import pandas as pd
    # data = pd.DataFrame(records, columns=cursor.column_names)

    # data_preprocessed = data[['order_date','basket']]
    # data_preprocessed.order_date = pd.to_datetime(data_preprocessed.order_date)
    # return data, data_preprocessed

with st.spinner('Loading Data...'):
    data, data_preprocessed = get_data_from_database()

# st.write("Sample Data:")
# st.write(data.head())

# st.write("Data getting Preprocessed:")
# st.write(data_preprocessed.head())

# %%
@st.cache
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

    return items

with st.spinner('Preprocessing Data...'):
    items = preprocess(data_preprocessed)

# st.write("Preprocessed Data:")
# st.write(items.head())

# %%
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

# %%
categorized_data = pd.DataFrame(items.values,index=items.index,columns=pd.MultiIndex.from_tuples(cat_lst,names=['cat1', 'cat2','items']))
categorized_data.sort_index(axis=1,level=['cat1','cat2','items'],inplace=True)
categorized_data.head()

# %%
items_data = categorized_data.sum(axis=1,level='items') # items data
category_data = categorized_data.sum(axis=1,level='cat2') # category wise items data
high_level_data = categorized_data.sum(axis=1,level='cat1') # for high level analysis


# %%
categories = sorted(list(set(categorized_data.columns.get_level_values(0))))
st.sidebar.markdown("## Select Max Lags and Duration of Prediction")
select_category = st.sidebar.selectbox('Select Category', categories)
sub_categories = sorted(list(set(categorized_data[select_category].columns.get_level_values(0))))
select_subcategory = st.sidebar.selectbox('Select SubCategory', sub_categories)
items_list = sorted(list(set(categorized_data[select_category][select_subcategory].columns.get_level_values(0))))
select_item = st.sidebar.selectbox('Select Item', items_list)

# if select_category:
#     sub_categories = set(categorized_data[select_category].columns.get_level_values(0))
#     select_subcategory = st.sidebar.selectbox('Select SubCategory', sub_categories)
# else:
#     select_category = categories[0]
#     sub_categories = set(categorized_data[select_category].columns.get_level_values(0))
#     select_subcategory = st.sidebar.selectbox('Select SubCategory', sub_categories)

# %%
import plotly.figure_factory as ff

st.write(select_category + " Sales:")
st.area_chart(high_level_data[select_category].resample('M').mean())

st.write(select_subcategory + " Sales:")
st.area_chart(category_data[select_subcategory].resample('M').mean())

st.write(select_item + " Sales:")
st.area_chart(categorized_data[select_category][select_subcategory][select_item].resample('M').mean())


# %%
# Function Definitions

def adf_test(ts, signif=0.05):
    dftest = adfuller(ts, autolag='AIC')
    adf = pd.Series(dftest[0:4], index=['Test Statistic','p-value','# Lags','# Observations'])
    for key,value in dftest[4].items():
       adf['Critical Value (%s)'%key] = value
    print (adf)
    
    p = adf['p-value']
    if p <= signif:
        print(f" Series is Stationary")
    else:
        print(f" Series is Non-Stationary")

@st.cache
def VAR_model_testing(df,max_lags,nobs):
    df_train, df_test = df[0:-nobs], df[-nobs:]
    model = VAR(df_train)
    results = model.fit(maxlags= max_lags, ic='aic')
    results.summary()

    # forecasting
    lag_order = results.k_ar
    predictions = results.forecast(df_train[-lag_order:].values, nobs)

    # plotting
    results.plot_forecast(nobs)

    df_forecast = pd.DataFrame(predictions, index=df.index[-nobs:], columns=df.columns+'_forecasted')

    rmse = math.sqrt(mean_squared_error(df_test.values, df_forecast.values))
    print("The root mean squared error is {}.".format(rmse))

    # fig, axs = plt.subplots()
    axs = df_test.plot(subplots=True)
    df_forecast.plot(subplots=True,ax = axs,colormap='gist_rainbow_r')

    # Impulse response
    irf = results.irf(nobs)
    irf.plot(orth=False)
    irf.plot_cum_effects(orth=False)

    # Evaluation
    fevd = results.fevd(5)
    fevd.summary()
    fevd.plot()

    return df_forecast, results

@st.cache
def VAR_model(df,max_lags,nobs):
    model = VAR(df)
    results = model.fit(maxlags= max_lags, ic='aic')
    results.summary()

    # forecasting
    lag_order = results.k_ar
    predictions = results.forecast(df[-lag_order:].values, nobs)

    # plotting
    results.plot_forecast(nobs)

    idx = pd.date_range(df.index[-1], periods=nobs, freq="M")
    df_forecast = pd.DataFrame(predictions, index=idx, columns=df.columns+'_forecasted')

    return df_forecast, results


@st.cache
def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:  
        print(col)      
        # Roll back 2nd Diff
        # if second_diff:
        #     df_fc[str(col)+'_1d'] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
        # # Roll back 1st Diff
        # df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
        df_fc[str(col)+'_forecasted'] = df_train[col].iloc[-1] + df_fc[str(col)+'_forecasted'].cumsum()
    return df_fc


# %%
nobs = 36
max_lag = 6
df = categorized_data[select_category][select_subcategory].resample('M').mean()
# df = categorized_data.Fruits.Apple.sum(axis=1,level='items')
df_forecast, results = VAR_model(df,max_lag,nobs)
# st.pyplot(fig, clear_figure=True)
# df_results = invert_transformation(df, df_forecast, second_diff=False)
df_results = df_forecast

# %%
st.write(select_item + " Forecasting:")
st.area_chart(df_forecast[select_item+'_forecasted'].resample('M').mean())

# %%
# combined_data = np.concatenate((df[select_item][0:-nobs].values,df_results[select_item+'_forecasted'].values))
# combined_index = np.concatenate((df[0:-nobs].index,df_results.index))
# combined_df = pd.DataFrame(combined_data,index=combined_index)
# st.area_chart(combined_df.resample('M').mean())

# %%
combined_data = np.concatenate((df[select_item].values,df_results[select_item+'_forecasted'].values))
combined_index = np.concatenate((df.index,df_results.index))
combined_df = pd.DataFrame(combined_data,index=combined_index)
st.area_chart(combined_df.resample('M').mean())
