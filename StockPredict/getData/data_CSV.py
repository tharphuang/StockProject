import tushare as ts

ts.set_token('b6fd53e9955f0cbb04d0ffb3610db2e09c645fea62044a8893f89090')
pro = ts.pro_api()
data=pro.query('daily',ts_code='601988.SH',start_date='20150101',end_date='20190524')#,end_date='20190412'
data.to_csv('/Users/hzp/Desktop/StockPredict/getData/601988.csv',columns=['trade_date','open','low','close','pre_close','change','high'])
'''
data2=pro.query('daily',ts_code='600519.SH',start_date='20150101',end_date='20190522')#,end_date='20190412'
data2.to_csv('/Users/hzp/Desktop/StockPredict/getData/600519.csv',columns=['trade_date','open','low','close','pre_close','change','high'])
'''
