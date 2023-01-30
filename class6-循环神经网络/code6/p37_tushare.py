import tushare as ts

df1 = ts.get_k_data('600519', ktype='D', start='2010-04-26', end='2020-04-26')
# 更改第一个参数：六位股票代码，下载需要的股票历史数据

datapath1 = "./SH600519.csv"
df1.to_csv(datapath1)