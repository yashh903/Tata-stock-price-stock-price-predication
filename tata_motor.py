#import libaries
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from datetime import datetime, timedelta

#importing dataset and data cleaning
df=pd.read_csv(r'C:\Users\YASH\Desktop\pandas\TATAMOTORS.NS.csv')
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
df=df.dropna()
df['Date']=pd.to_datetime(df['Date'])


#EDA
#Fundamental Analysis 
sns.lineplot(data=df,y=df['Close'],x=df['Date'],color='Green')
plt.xlabel('Year')
plt.ylabel('Price')
plt.title('Tata Motors Price')


df['year'] = pd.DatetimeIndex(df['Date']).year
df['Market cap']=df['Open']*df['Volume']
sns.lineplot(data=df,y=df['Market cap'],x=df['year'])
plt.xticks(rotation=90)
plt.title('Market capitalization')


df['vol'] = (df['Close'] / df['Close'].shift(1)) - 1
sns.lineplot(data=df,x=df['year'],y=df['vol'])
plt.title('Volatility')


df['Cumulative Return'] = (1 + df['vol']).cumprod()
sns.barplot(data=df,x=df['year'],y=df['Cumulative Return'])
plt.xticks(rotation=90)
plt.title('Cumulative Return')


df['Return'] = df['Close'].pct_change()
yearly_returns = df.groupby(['year'])['Return'].mean().reset_index()
sns.barplot(data=yearly_returns,x='year',y='Return')
plt.xticks(rotation=90)
plt.title('Average Yearly Returns')


#Technical Analysis
df['MA for 10 days'] = df['Open'].rolling(10).mean()
df['MA for 20 days'] = df['Open'].rolling(20).mean()
df['MA for 50 days'] = df['Open'].rolling(50).mean()
df['MA for 100 days'] = df['Open'].rolling(100).mean()
truncated_data = df.truncate()
truncated_data[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days', 'MA for 100 days']].plot(subplots=False)
plt.title('Moving Average')
plt.xlabel('Day')
plt.ylabel('Price')


rolling_mean = df['Close'].rolling(window=20).mean()
rolling_std = df['Close'].rolling(window=20).std()
upper_band = rolling_mean + (rolling_std * 2)
lower_band = rolling_mean - (rolling_std * 2)
fig, ax = plt.subplots()
ax.plot(df.index, df['Close'], label='Close')
ax.plot(rolling_mean.index, rolling_mean, label='Rolling Mean')
ax.fill_between(rolling_mean.index, upper_band, lower_band, alpha=0.4, color='gray', label='Bollinger Bands')
ax.legend()
plt.title('Bollinger Bands')
plt.xlabel('Day')
plt.ylabel('Price')
plt.show()

sns.heatmap(df.corr())
plt.title("Correlation Analysis")


#Model Evaluation and Prediction
x=df.drop(['Date','Close','Adj Close'],axis=1)
y=df.Close

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)

lm=LinearRegression()
lm.fit(x_train,y_train)
pred=lm.predict(x_test)

r2_score(y_test,pred)

df_pred = pd.DataFrame(y_test.values,columns=['Actual'],index=y_test.index)
df_pred['predicated']=pred
sns.lineplot(df_pred[['Actual','predicated']])

coefficients = lm.coef_
intercept = lm.intercept_


equation = f"Close = {intercept:.4f} + {coefficients[0]:.4f} * Open + {coefficients[1]:.4f} * High + {coefficients[2]:.4f} * Low + {coefficients[3]:.4f} * Volume"
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

for i, col in enumerate(['Open', 'High', 'Low', 'Volume']):
    sns.scatterplot(x=col, y='Close', data=df, ax=axes[i//2, i%2], label='Actual Data')
    sns.regplot(x=col, y='Close', data=df, ax=axes[i//2, i%2], scatter=False, color='red', label='Regression Line')

fig.suptitle('Linear Regression: Feature vs Close')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()




#forecasting the stock price
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

mu = df['Close'].pct_change().mean() 
sigma = df['Close'].pct_change().std()  
dt = 1  
num_simulations = 100
num_future_days = 90

future_data = pd.DataFrame(index=pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=num_future_days, freq='D'))

for i in range(num_simulations):
    prices = [df['Close'].iloc[-1]]

    for day in range(1, num_future_days + 1):
        daily_return = np.random.normal(mu * dt, sigma * np.sqrt(dt))
        price_today = prices[-1] * (1 + daily_return)
        prices.append(max(0, price_today))  

    future_data[f'Simulation_{i+1}'] = prices[1:]

future_data.plot(legend=False, alpha=0.5)
plt.title('Monte Carlo Simulations of Future Stock Prices')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.show()

simulation_metrics = pd.DataFrame(index=future_data.columns)
simulation_metrics['Mean_Return'] = future_data.pct_change().mean()
simulation_metrics['Final_Value'] = future_data.iloc[-1]
simulation_metrics['Volatility'] = future_data.pct_change().std()
simulation_metrics['Sharpe_Ratio'] = simulation_metrics['Mean_Return'] / simulation_metrics['Volatility']

best_simulation = simulation_metrics['Sharpe_Ratio'].idxmax()

future_data[best_simulation].plot(label='Best Simulation', color='green', linewidth=2)
plt.title('Forecast')
plt.xlabel('Days')
plt.ylabel('Stock Price')
plt.legend()
plt.show()






















