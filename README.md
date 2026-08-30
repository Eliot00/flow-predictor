# flow-predictor

模拟店铺客流预测。

- [x] Random Forest
- [x] Hist Gradient Boosting
- [x] LSTM

## 特征

- 日期
- 天气（晴雨阴）
- 温度
- 油价
- 店铺面积
- 地理位置
- 主营类目

## 结果记录

### Random Forest

```plaintext
passby_visit: MSE=2946555.71, R²=0.33
entering_people: MSE=3361.64, R²=0.92
dwell_people: MSE=1387.15, R²=0.93
served_people: MSE=201.27, R²=0.88
```

### Hist Gradient Boosting

```plaintext
passby_visit: MSE=2464725.43, R²=0.44
entering_people: MSE=3205.78, R²=0.92
dwell_people: MSE=1620.49, R²=0.92
served_people: MSE=357.75, R²=0.80
```

### LSTM

```plaintext
Epoch 10/50 - Loss: 4548.6006
Epoch 20/50 - Loss: 3013.7742
Epoch 30/50 - Loss: 2910.1829
Epoch 40/50 - Loss: 2852.9103
Epoch 50/50 - Loss: 2772.9515
passby_visit: MSE=11451.92 R2=0.66
entering_people: MSE=302.34 R2=0.77
dwell_people: MSE=144.54 R2=0.74
served_people: MSE=33.84 R2=0.69
```

## 问题记录

- Fine-tune 新增店铺，LSTM上没有历史数据
