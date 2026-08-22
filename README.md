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
MSE: 14262.39
R2: 0.42
         feature  coefficient
0           area    46.109304
2       latitude    38.968216
5        weekday    26.171061
3    temperature    18.976015
7    day_of_year    13.300548
1      longitude    -0.002664
9  category_code    -0.204621
4      oil_price    -0.301439
6          month   -17.732516
8   weather_code   -71.657876
```
