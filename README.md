# Covid19-analysis-and-forecasting
Analysis of Covid-19 cases and forecasting the spread for India

### Dataset : covid_19_data.csv (Data Available till 26th-March-2020)

SNo	ObservationDate	Province/State	Country/Region	Last Update	Confirmed	Deaths	Recovered

                                                         
| __Column name__    | __Detail__                                                 |
|--------------------|------------------------------------------------------------|
| SNo                |  Serial Number                                             |
| ObservationDate    |  Date of Case observed                                     |
| Province/State     |  State                                                     |
| Country/Region     |  Country                                                   |
| Last Update        |  Last Updation of that Case                                |
| Confirmed          |  Confirmation of Cases for that Day-State                  |
| Deaths             |  Deaths of for that Day-State                              |
| Recovered          |  Recoveries for that Day-State                             |

### Top 10 Countries with corona infected cases 

![top10](https://github.com/yatinkode/Covid19-analysis-and-forecasting/blob/master/images/top10.gif)

### Interactive World based distribution of Corona confirmed cases
![top10](https://github.com/yatinkode/Covid19-analysis-and-forecasting/blob/master/images/geomap.gif)

### Forecasting Corona cases for next 10 days in India (Core python timeseries)
![top10](https://github.com/yatinkode/Covid19-analysis-and-forecasting/blob/master/images/forecast.png)

| 27-03-20 | 28-03-20 | 29-03-20 | 30-03-20 | 31-03-20 | 1-04-20 | 2-04-20 | 3-04-20 | 4-04-20 | 5-04-20 |
|----------|----------|----------|----------|----------|---------|---------|---------|---------|---------|
| 864.80   | 1071.76  | 1249.90  | 1470.40  | 1709.48  | 1987.10 | 2279.99 | 2677.18 | 3098.26 | 3690.31 |

### Forecasting Corona cases for next 10 days in India (Facebook prophet timeseries)
![top10](https://github.com/yatinkode/Covid19-analysis-and-forecasting/blob/master/images/forecastfb.png)

| 27-03-20 | 28-03-20 | 29-03-20 | 30-03-20 | 31-03-20 | 1-04-20 | 2-04-20 | 3-04-20 | 4-04-20 | 5-04-20 |
|----------|----------|----------|----------|----------|---------|---------|---------|---------|---------|
| 718.71   | 890.26   | 973.72   | 1107.65  | 1093.35  | 1213.86 | 1238.81 | 1150.06 | 1327.57 | 1362.98 |

