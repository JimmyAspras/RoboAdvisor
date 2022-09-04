# RoboAdvisor

## Introuction

This is a relatively simple project using data to build a roboadvisor and compare returns against a benchmark rate of return.

Data for this project was chosen and downloaded from Wharton Research Data Services: https://wrds-www.wharton.upenn.edu/. This was done as part of a course taken in Summer 2021 complete with prompts and analysis. Credit must be given to my professor, Dr. Wei Jiao, for much code and instruction included here.

This project follows a series of prompts to to build a roboadvisor to beat market return.

## Preparing the Data

**Select three ETFs and download their price information from Yahoo Finance. I chose VOOG S&P 500 growth, VYM Large cap high dividend yield, and BLV Long term bond ETFs.**

### Libraries

```python
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import minimize
plt.rcParams['figure.figsize'] = [20, 15]
```

A start date of 1/1/2011 was chosen because this is the first year for which data for all three ETFs is available
```python
start_date = datetime(2011, 1, 1) #set start date
end_date = datetime(2021, 6, 30) #set end date
```

Create dataframe for the three ETFs. Make vectors for Year and Month.
```python
etf1 = yf.download(['VOOG','VYM','BLV'], start = start_date, end = end_date)['Adj Close']
etf1.reset_index(inplace=True)
etf1['Year']=etf1['Date'].dt.year
etf1['Month']=etf1['Date'].dt.month
etf1.rename(columns={'VOOG':'fund1','VYM':'fund2','BLV':'fund3'},inplace=True)
etf1
```

Output

<img width="456" alt="image" src="https://user-images.githubusercontent.com/72087263/188316663-37ef784b-3ce4-441c-bc71-52097d5201d0.png">

Sort values by date
```python
etf1.sort_values(by=['Date'], inplace=True)
```

Calculate the monthly returns of the three ETFs you select.

```python
etf1[['fund1_ret_d','fund2_ret_d','fund3_ret_d']]=etf1[['fund1','fund2','fund3']].pct_change()
etf1[['fund1_ret_d+1','fund2_ret_d+1','fund3_ret_d+1']]=etf1[['fund1_ret_d','fund2_ret_d','fund3_ret_d']]+1
etf2=etf1[['fund1_ret_d+1','fund2_ret_d+1','fund3_ret_d+1','Year','Month']].groupby(['Year','Month']).prod()
etf2[['fund1_ret_m','fund2_ret_m','fund3_ret_m']]=etf2[['fund1_ret_d+1','fund2_ret_d+1','fund3_ret_d+1']]-1
etf3=etf2[['fund1_ret_m','fund2_ret_m','fund3_ret_m']]
etf3
```

Output

<img width="361" alt="image" src="https://user-images.githubusercontent.com/72087263/188316705-06888128-7501-43f9-866e-cf7f313e924e.png">

**Pick a target annualized volatility of the portfolio of the three ETFs. Search  and  report  the weights on the three ETFs that maximize the average return of the portfolio of the three ETFs for your target volatility level. (search_weight function in the example code can be used to search the weights) If you pick a too high or too low target volatility, the search might be unsuccessful. Then, please adjust your target volatility and make sure the search is successful.**

Calculate current return and volatility of equally weighted portfolio
```python
returns=etf3
weight=[0.333,0.333,0.333]
returns.multiply(weight)
returns.multiply(weight).sum(axis=1)
def pret(weight):
    pret1=returns.multiply(weight).sum(axis=1)
    pret1_mean=pret1.mean()
    return pret1_mean
pret(weight)
```

Return

```python
0.010089847632325401
```

```python
pret(weight)
def pvol(weight):      
    pret1=returns.multiply(weight).sum(axis=1)
    pret1_vol=pret1.std()*np.sqrt(12)
    return pret1_vol
pvol(weight)
```

Volatility

```python
0.08644663420249178
```

Create a function to find optimal volume and weight and calculate the returns.
```python
target_vol=[0.1]
no_fund=3
weight=[0.25,0.25,0.5]
returns=etf3

def search_weight(weight, returns, target_vol, no_fund):

    def pret(weight):
        pret1=returns.multiply(weight).sum(axis=1)
        pret1_mean_flip=-pret1.mean()
        return pret1_mean_flip

    def pvol(weight,target_vol):
        pret1=returns.multiply(weight).sum(axis=1)
        pret1_vol=pret1.std()*np.sqrt(12)-target_vol
        return pret1_vol

    def sumweight(weight):
        return weight.sum()-1

    solve1=minimize(pret, weight,constraints=({"fun": pvol, "type": "eq", 'args': target_vol},{"fun": sumweight,"type": "eq"}), bounds=[(0,1)]*no_fund)

    weight_select=solve1.x
    portfolio_ret=-solve1.fun*12
    success=solve1.success

    return portfolio_ret, weight_select, success;
```

Input starting values

```python
weight=[0,0,0]
returns=etf3
target_vol=[0.09]
no_fund=3
```

Run functions and display return

```python
portfolio_ret,weight_select,success=search_weight(weight, returns, target_vol, no_fund)
portfolio_ret
```

Return
```python
0.12623419404516698
```

Display weights
```python
weight_select
```

```python
array([0.4248134 , 0.26702736, 0.30815924])
```

**As you progress in your career, your income is likely to increase over time. Thus, you could put more money each month into your investment portfolio. Please select a growth rate for your monthly payment and revise the accountbalance function to incorporate the monthly payment growth into the function. (For example, if you think you will put 0.1% more money monthly overtime and the first monthly payment is $200, then your second monthly payment is $200 * (1+0.1%) and the Nth monthly payment is $200 * (1+0.1%) ^N.)**

```python
age_current = 31
age_retire=65
monthlypayment=500
no_simulation=100
annualfee=0.0035
pmtincrease=0.005
#pmtincrease is a variable added for expected increase in monthly contributions
#portfolio_ret and target_vol are values recorded previously

#The function to calculate accountbalance        
def accountbalance(age_current, age_retire, monthlypayment, no_simulation, annualfee, pmtincrease, portfolio_ret, target_vol):

    no_month=(age_retire-age_current)*12

#simulate returns using random numbers from normal distribution using portfolio return and target vol from previous calculation
    pret_sim1=np.random.normal(portfolio_ret/12,target_vol/np.sqrt(12),size=(no_month,no_simulation))

#take into account advisory fees
    pret_sim2=pret_sim1-annualfee/12

#simulate account balance over time
##(For  example,  if  you  think  you  will  put  0.1%  more  money  monthly overtime  and  the  
#first  monthly  payment  is  $200,  then  your  second  monthly  payment  is  $200  * (1+0.1%) 
#and the Nth monthly payment is $200 * (1+0.1%) ^N.)
    value=0
    balance=[]
    
    for i in range (no_month):
        value=(value+monthlypayment*(1+pmtincrease)**i)*(1+pret_sim2[i,:])
        balance.append(value)

    balance1=pd.DataFrame(balance)

    balance1['month_no']=balance1.index+1
    
    #Reshape the balance1 file
    balance2=pd.melt(balance1, id_vars=['month_no'], var_name='sim_no', value_name='balance')
    
#We set the median account balance in each month as the balance under normal market condition
#With 50% chance, the account balance is at least this amount
    normal1=balance2[['month_no','balance']].groupby(['month_no']).quantile(0.5)
    normal1['balance_m']=normal1['balance']/1000000
#We set the 10th percentile account balance in each month as the balance under weak market condition
#With 90% chance, the account balance is at least this amount
    weak1=balance2[['month_no','balance']].groupby(['month_no']).quantile(0.1)
    weak1['balance_m']=weak1['balance']/1000000
    
    return normal1,weak1;
```

```python
#accountbalance(age_current, age_retire, monthlypayment, no_simulation, annualfee, pmtincrease, portfolio_ret, target_vol)
age_current=31
age_retire=60
monthlypayment=500
no_simulation=1000
annualfee=0.0035
pmtincrease=0.005

normal1,weak1=accountbalance(age_current, age_retire, monthlypayment, no_simulation, annualfee, pmtincrease, portfolio_ret, target_vol)
normal1
```

Output

<img width="245" alt="image" src="https://user-images.githubusercontent.com/72087263/188318022-0c2216d9-0bec-4ac4-853f-1a3d475a4fd6.png">

```python
weak1
```

Output

<img width="245" alt="image" src="https://user-images.githubusercontent.com/72087263/188318061-85e5c35a-0956-454e-bba3-b41ba5efdd01.png">

**Generate and report the figure showing your account balance over time under normal and weak market conditions after incorporating the monthly payment growth. The calculation should be based on the three ETFs you select in question 1 and the target volatility and weights in question 3. (Please free to select the values for all other relevant parameters, such as current age, retirement age, monthly payment, number of simulations, etc.)**

```python
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
plt.rcParams['ytick.left'] = plt.rcParams['ytick.labelleft'] = False
plt.plot(normal1['balance_m'],label="normal market")
plt.plot(weak1['balance_m'],label='weak market')
plt.xlabel("No. of months", size=15)
plt.title("Account Balance ($million)",size=36)
plt.xticks(size=22)
plt.yticks(size=22)
plt.legend(fontsize=22)
```

Plot of returns

![image](https://user-images.githubusercontent.com/72087263/188318121-1444c7ea-23fe-4ef4-bf89-c868d488074e.png)

## Conclusion

In this example from start to finish, we see that using data from the funds VOOG, VYM, and BLV with a start date of 1/1/11 and finish date of 6/30/21, there is a: Portfolio return of 0.010089847632325401 and portfolio volatility of 0.08644663420249178. Considering and equal weight portfolio with a target volatility of 0.09, there is a portfolio return of 0.12623419404516698 with the following weights for each fund:([0.4248134 , 0.26702736, 0.30815924])

Using the following inputs in the account balance function:
```
age_current=31
age_retire=60
monthlypayment=500
no_simulation=1000
annualfee=0.0035
pmtincrease=0.005
```
We see returns approaching $2.7 million and $1.7 million in normal and weak markets, respectively.
