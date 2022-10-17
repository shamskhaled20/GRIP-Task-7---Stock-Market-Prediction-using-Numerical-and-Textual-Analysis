#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import datetime
from datetime import date 
import yfinance as yf
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[38]:


pip install pandas_datareader


# In[39]:


#Web data reader is extension of pandas library to communicate with frequently updating data
import pandas_datareader.data as web
from pandas import Series, DataFrame
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime(2019, 12, 31)

df = web.DataReader("MSFT", 'yahoo', start, end)
df.tail()


# 1_ Data Analysis
# 

# In[40]:


df.head()


# In[41]:


df.isnull().sum()


# In[42]:


close_px = df['Adj Close']
mavg = close_px.rolling(window = 100).mean()
mavg.tail(10)


# In[43]:


df.shape


# 2_ Data Visualization
# 

# In[44]:


#Closing Stock
df['Close'].hist()


# In[45]:


df['Close'].plot()
plt.xlabel("Date")
plt.ylabel("Close")


# In[46]:


df['Close'].plot(style='.')
plt.title("Scatter plot of Closing Price")
plt.title('Scatter plot of Closing Price',fontsize=20)
plt.show()


# In[47]:


#Test for staionarity
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    #Determing rolling statistics
    rolling_mean = timeseries.rolling(12).mean()
    rolling_std = timeseries.rolling(12).std()
    #Plot rolling statistics:
    plt.plot(timeseries, color='Black',label='Original')
    plt.plot(rolling_mean, color='Green', label='Rolling Mean')
    plt.plot(rolling_std, color='Red', label = 'Rolling Standard Deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation',fontsize=20)
    plt.show(block=False)
    
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    # output for dft will give us without defining what the values are.
    #hence we manually write what values does it explains using a for loop
    output = pd.Series(adft[0:4],index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key,values in adft[4].items():
        output['critical value (%s)'%key] =  values
    print(output)
    
test_stationarity(df['Close'])


# In[48]:


from pylab import rcParams
rcParams['figure.figsize'] = 10, 9
df_log = np.log(df.Close)
moving_avg = df.Close.rolling(12).mean()
std_dev = df.Close.rolling(12).std()
plt.legend(loc='best')
plt.title('Moving Average',fontsize=20)
plt.plot(std_dev, color ="Blue", label = "Standard Deviation")
plt.plot(moving_avg, color="Green", label = "Mean")
plt.legend()
plt.show()


# In[49]:


#Split data into train and training set
train_data, test_data = df_log[3:int(len(df_log)*0.9)], df_log[int(len(df_log)*0.9):]
plt.figure(figsize=(10,9))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(df_log, 'green', label='Train data')
plt.plot(test_data, 'blue', label='Test data')
plt.legend()


# In[50]:


pip install pmdarima


# In[51]:


import statsmodels
from statsmodels import compat
from statsmodels.compat import pandas
from pmdarima.arima import auto_arima

auto_arima_model = auto_arima(train_data, start_p=0, start_q=0,
                      test='adf',       # use adftest to find             optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
print(auto_arima_model.summary())


# In[52]:


auto_arima_model.plot_diagnostics()
plt.show()


# In[53]:


import statsmodels.api as sm
model_arima=sm.tsa.arima.ARIMA(train_data,order=(2,1,0))
fit_model=model_arima.fit()
print(fit_model.summary())


# In[54]:


fc, se, conf = fit_model.forecast(504, alpha=0.05) 
fc_series = pd.Series(fc, index=test_data.index)
low = pd.Series(conf[0, :], index=test_data.index)
up = pd.Series(conf[:, 1], index=test_data.index)


# In[ ]:


plt.figure(figsize=(10,9), dpi=100)
plt.plot(train_data, label='Training Set')
plt.plot(test_data, color = 'Black', label='Actual Stock Price')
plt.plot(fc_series, color = 'Green',label='Predicted Stock Price')
plt.fill_between(low.index, low, up, color='k', alpha=.10)
plt.title('S&P BSE SENSEX Stock Price Prediction',fontsize=20)
plt.xlabel('Year')
plt.ylabel('Stock Price')
plt.legend(fontsize=10,loc='upper left')
plt.show()


#  Analyze stocks of various IT giants like Microsoft, Apple, Amazon, Google ,IBM.
# 

# In[ ]:


dfcomp = web.DataReader(['MSFT' , 'AAPL' , 'AMZN' , 'GOOG' , 'IBM'], 'yahoo', start=start, end=end)['Adj Close']
dfcomp.head()


# In[ ]:


retscomp = dfcomp.pct_change()
corr = retscomp.corr()
retscomp.head(10)


# Using Heat Maps to visualize correlation range of various stocks
# 

# In[ ]:


plt.imshow(corr, cmap = 'hot', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns);


# In[ ]:


plt.scatter(retscomp.mean(), retscomp.std())
plt.xlabel('Expected returns')
plt.ylabel('Risk')
for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
  plt.annotate(label, xy = (x, y), xytext = (20, -20),
               textcoords = 'offset points', ha = 'right', va = 'bottom',
               bbox = dict(boxstyle = 'round, pad = 0.5', fc  = 'yellow',
                           alpha = 0.5), arrowprops = dict(arrowstyle = '->',
                                                           connectionstyle = 'arc3,rad = 0'))


# 2. Textual Analysis
# 

# In[ ]:


import pandas as pd
import numpy as np 
df1 = pd.read_csv('Downloads/india-news-headlines.csv')
df1.head(10)


# A) Data Analysis
# 

# In[ ]:


df1.tail(10)


# In[ ]:


df1.shape


# In[ ]:


df1.isnull().sum()


# In[ ]:


df1.info()


# In[ ]:


df1.describe()


# In[ ]:


df1.max()


# In[ ]:


df1.min()


# In[ ]:


df1['headline_category'].unique()


# In[ ]:


df1.corr()


# In[ ]:


df1.shape


# B) Data Visualization
# 

#  EDA (Exploratry data analysis) using NLP and NLTK tools
# 

# In[ ]:


sns.set_palette('viridis')
sns.pairplot(df1)
plt.show()


# In[ ]:


df1['headline_text'].value_counts()


# In[ ]:


df1['headline_category'].value_counts()


# In[ ]:


df1['headline_text'].str.len().hist()
plt.show()


# In[ ]:


def basic_clean(text):
        wnl = nltk.stem.WordNetLemmatizer()
        stopwords = nltk.corpus.stopwords.words('english')
        words = re.sub(r'[^\w\s]', '', text).split()
        return [wnl.lemmatize(word) for word in words if word not in stopwords]


# In[ ]:


words = basic_clean(''.join(str(df1['headline_text'].tolist())))
words[:10]


#  N-Gram Analysis
# 

# a) Unigram Analysis
# 

# In[ ]:


import pandas as pd
words_unigram_series = (pd.Series(nltk.ngrams(words, 1)).value_counts())[:20]


# In[ ]:


words_unigram_series.sort_values().plot.barh(color='lightcoral', width=.9, figsize=(12, 8))
plt.title('20 Most Frequently Occuring Unigrams - India News Headlines')
plt.ylabel('Bigram')
plt.xlabel('# of Occurances')


# b) Bigram Analysis
# 

# In[ ]:


words_bigrams_series = (pd.Series(nltk.ngrams(words, 2)).value_counts())[:20]


# In[ ]:


words_bigrams_series.sort_values().plot.barh(color='thistle', width=.9, figsize=(12, 8))
plt.title('20 Most Frequently Occuring Bigrams - India News Headlines')
plt.ylabel('Bigram')
plt.xlabel('# of Occurances')


# c) Trigram Analysis
# 

# In[ ]:


words_trigrams_series = (pd.Series(nltk.ngrams(words, 3)).value_counts())[:20]
words_trigrams_series.sort_values().plot.barh(color='darksalmon', width=.9, figsize=(12, 8))
plt.title('20 Most Frequently Occuring Trigrams - India News Headlines')
plt.ylabel('Trigram')
plt.xlabel('# of Occurances')


# In[ ]:




