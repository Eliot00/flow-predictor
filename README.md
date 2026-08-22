# flow-predictor

模拟店铺客流预测。

- [x] Linear Regression
- [ ] MLP
- [ ] LSTM

## 特征

- 日期
- 天气（晴雨阴）
- 温度
- 油价
- 店铺面积
- 地理位置
- 主营类目

## 结果记录

### 线性回归

特征系统基本和生成假数据的逻辑吻合了

```
passby_visit: MSE=17052.43, R²=0.34
entering_people: MSE=596.09, R²=0.18
dwell_people: MSE=256.56, R²=0.25
served_people: MSE=62.52, R²=0.33

系数矩阵（行=目标，列=特征）:
                      area  longitude   latitude  ...  day_of_year  weather_code  category_code
passby_visit     53.807875  -0.982862  35.900892  ...    16.206073    -79.192350       4.267923
entering_people  11.939922  -1.045672   7.153583  ...     3.342367     -6.026688      21.178417
dwell_people      8.641613   0.067767   4.182763  ...     1.905367     -3.647974       4.197329
served_people     3.651417   0.123250   1.830584  ...     0.819806     -1.690770      -2.574629
```
