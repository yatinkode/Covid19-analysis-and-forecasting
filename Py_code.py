################################################################
#Covid-19 Data Analysis and Forecasting
################################################################

############### Load libraries ################################
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

import statsmodels.api as sm
import itertools

#libraries for plotting
import json
from matplotlib import pyplot as plt
import seaborn as sns
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar
from bokeh.palettes import brewer
import geopandas as gpd

#libraries for facebook prophet
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric

############### Load dataset ##################################
data =  pd.read_csv("C:\\Users\\kode surendra aba\\Desktop\\Data science\\novel-corona-virus-2019-dataset\\covid_19_data.csv")

data.head()

############### Data Cleaning #################################
data = data.rename({'ObservationDate': 'Date', 'Province/State': 'State','Country/Region':'Country'}, axis=1) 

data.drop(['Last Update', 'SNo','State'], axis=1, inplace=True)

data = data.replace(to_replace =["Mainland China"],  value ="China") 
data = data.replace(to_replace =["Republic of the Congo","Congo (Brazzaville)","Congo (Kinshasa)"],  value ="Congo") 
data = data.replace(to_replace =["Cabo Verde"],  value ="Cape Verde") 
data = data.replace(to_replace =["Bahamas, The"],  value ="The Bahamas")
data = data.replace(to_replace =["Gambia, The"],  value ="The Gambia")
data = data.replace(to_replace =["('St. Martin',)"],  value ="St. Martin")
data = data.replace(to_replace =["Republic of Ireland"],  value ="Ireland")
data = data.replace(to_replace =["Reunion"],  value ="France")
data = data.replace(to_replace =["occupied Palestinian territory"],  value ="Palestine")
data = data.replace(to_replace =["Holy See"],  value ="Vatican City")
data = data.replace(to_replace =[" Azerbaijan"],  value ="Azerbaijan")
data = data.replace(to_replace =["Cayman Islands"],  value ="Channel Islands")
data = data.replace(to_replace =["Timor-Leste"],  value ="East Timor")

data['Date'] =  pd.to_datetime(data['Date'], format='%m/%d/%Y')

last_day = data['Date'].sort_values(ascending=False).head(1).reset_index(drop=True)

Final_data = data[data['Date'] == datetime.strptime(last_day[0].date().strftime('%Y-%m-%d'),"%Y-%m-%d")]

Final_data.head(10)

grouped = Final_data.groupby('Country',as_index=False)
Final_data = (grouped['Confirmed','Deaths','Recovered'].agg(np.sum)).sort_values(by="Confirmed",ascending=False)

df = pd.melt(Final_data.head(10), id_vars="Country", var_name="Type", value_name="Count")

plt.figure(figsize=(5000, 3000))
splot = sns.catplot(x="Count", y="Country", hue="Type", kind="bar", data=df,height=10)

############### Plotting Top 10 Countries ##############################

cc =  pd.read_csv("C:\\Users\\kode surendra aba\\Desktop\\Data science\\novel-corona-virus-2019-dataset\\countrycode.csv")

Final_data = cc.merge(Final_data, on='Country', how='left')

shapefile = 'C:\\Users\\kode surendra aba\\Desktop\\Data science\\novel-corona-virus-2019-dataset\\ne_110m_admin_0_countries\\ne_110m_admin_0_countries.shp'#Read shapefile using Geopandas
gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]#Rename columns.
gdf.columns = ['Country', 'Code', 'geometry']
gdf.head()

print(gdf[gdf['Country'] == 'Antarctica'])#Drop row corresponding to 'Antarctica'
gdf = gdf.drop(gdf.index[159])

Final_data = gdf.merge(Final_data, on='Code', how='left')

Final_data.rename(columns = {"Country_y": "Country"}, inplace = True) 

Final_data.fillna('No data', inplace = True)

Final_data = Final_data.drop(columns=['Country','Deaths','Recovered'])

merged_json = json.loads(Final_data.to_json())#Convert to String like object.
json_data = json.dumps(merged_json)

#################### Plotting cases on World map ##########################

#Input GeoJSON source that contains features for plotting.
geosource = GeoJSONDataSource(geojson = json_data)    #Define a sequential multi-hue color palette.
palette = brewer['OrRd'][8]   #Reverse color order so that dark blue is highest cases.
palette = palette[::-1]      #Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
color_mapper = LinearColorMapper(palette = palette, low = 0, high = 50000, nan_color = '#d9d9d9')#Define custom tick labels for color bar.
tick_labels = {'0': '0', '100': '100', '500':'500', '1000':'1000', '5000':'5000','10000':'10000', '50000': '>50000'}#Create color bar. 
color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,
border_line_color=None,location = (0,0), orientation = 'horizontal', major_label_overrides = tick_labels)#Create figure object.
p = figure(title = 'Corona confirmed cases as of {}'.format(last_day[0].date().strftime('%d-%b-%Y')), plot_height = 550 , plot_width = 900, toolbar_location = None)
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None

#Add patch renderer to figure. 
p.patches('xs','ys', source = geosource,fill_color = {'field' :'Confirmed', 'transform' : color_mapper},
          line_color = 'black', line_width = 0.25, fill_alpha = 1)#Specify figure layout.
p.add_layout(color_bar, 'below')#Display figure inline in Jupyter Notebook.
show(p)

######################### Forecasting future cases for India of Coronavirus #################
ind = data[data['Country']=='India'] #Choosing India data now
ind.head()

ind.drop(['Country', 'Deaths','Recovered'], axis=1, inplace=True)

ind = ind.set_index('Date')
ind.index

#Plotting the time series
ind.plot(figsize=(15, 6))

y = ind['Confirmed'].resample('D',level=0).mean()

decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

#Find best parameters for p d q
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

#ARIMA(1, 0, 1)x(1, 1, 1, 12)12 - AIC:272.66002216323875 is Best and lowest AIC

#fitting the best ARIMA model
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 0, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()
print(results.summary().tables[1])

results.plot_diagnostics(figsize=(16, 8))
plt.show()

#Validating forecasts
pred = results.get_prediction(start=pd.to_datetime('2020-01-30'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2020':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Confirmed Cases')
plt.legend()
plt.show()


y_forecasted = pred.predicted_mean
y_truth = y['2020-01-30':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error {} and Root Mean Squared Error is {}'.format(round(mse, 2),round(np.sqrt(mse), 2)))

pred_uc = results.get_forecast(steps=10)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Confirmed Cases')
plt.legend()
plt.show()

print(pred_uc.predicted_mean)
#2020-03-27     864.804356
#2020-03-28    1071.762124
#2020-03-29    1249.901131
#2020-03-30    1470.407147
#2020-03-31    1709.485385
#2020-04-01    1987.102183
#2020-04-02    2279.993056
#2020-04-03    2677.183613
#2020-04-04    3098.262067
#2020-04-05    3690.317032

################################### Facebook prophet timeseries ##########################

daily_train = ind.resample('D').sum()

#facebook prophet needs the names of the 2 columns to be ds and y respectively
daily_train['ds'] = daily_train.index
daily_train['y'] = daily_train.Confirmed
daily_train.drop(['Confirmed'],axis = 1, inplace = True)

m = Prophet(growth="logistic",seasonality_mode='multiplicative') #logistic - since sudden groeth at a point

daily_train['cap'] = 1500 #maximum capacity for time series based on business problem

m.fit(daily_train)
future = m.make_future_dataframe(periods=10)
future['cap'] = 1500
forecast = m.predict(future)

m.plot_components(forecast)

m.plot(forecast)

a = m.plot(forecast)

df_cv = cross_validation(m, initial='10 days', period='2 days', horizon = '5 days')
df_cv.head()

df_p = performance_metrics(df_cv)
df_p.head()


fig = plot_cross_validation_metric(df_cv, metric='mape')

forecast[['ds','yhat']].tail(10)

#2020-03-27   718.719249
#2020-03-28   890.265552
#2020-03-29   973.727510
#2020-03-30  1107.658426
#2020-03-31  1093.352717
#2020-04-01  1213.865651
#2020-04-02  1238.818136
#2020-04-03  1150.066958
#2020-04-04  1327.578203
#2020-04-05  1362.981341
