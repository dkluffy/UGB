# UGB - dev log

## TimeLine

- 2020/01/31 已经实现了大部分训练功能
  - tensorboard
  - save_weights,load_model,checkpoints，scheduler，。。。
  - 用dataset大大简化了数据生成问题
  - 训练了50 epoch,loss change from 2.1244 to 1.0398

```python
100/100 [==============================] - 86s 861ms/step - loss: 1.0488
Epoch 45/50
100/100 [==============================] - 86s 858ms/step - loss: 1.0475
Epoch 46/50
100/100 [==============================] - 90s 905ms/step - loss: 1.0451
Epoch 47/50
100/100 [==============================] - 81s 815ms/step - loss: 1.0443
Epoch 48/50
100/100 [==============================] - 86s 860ms/step - loss: 1.0425
Epoch 49/50
100/100 [==============================] - 85s 855ms/step - loss: 1.0412
Epoch 50/50
100/100 [==============================] - 86s 861ms/step - loss: 1.0398

"""
loss可能卡在1附近不动了， 用较大的LR和极小LR测试，LOSS都没能成功下来，可能还要调整数据集
暂时先放一下，还有别事情要做
"""
#TODO:
# 调整lr 和 optimizer，batch_size等
# 调整数据集
# 调整model,从目前训练来看，训练过的weights，貌似没有帮助
# 实现metric


```

## 目标

- 高精度（不受分辨率、大小影响）

- 不需要直接返回坐标，不需要端到端

- 模型可以简单训练

- oneshot:旧版利用opencv实现的几乎时zero-shot，但受限于精度

- 最少的人工标签

## 实现对比

### 分块对比法-tripletloss版

通过把X（截屏得到的图片）裁剪成N个固定大小(128,128,3)小块$x_i$，对比$x_i$和模板$T_i$ 的L2 距离。

#### 训练方法

- 双生网络，人脸认证的思想，对比$x_i,t_i$的encode l2距离
- from tensorflow_addons.losses import TripletSemiHardLoss
- 使用训练过的mobilenetv2权值
- 用noise和模板的部分截图$t_i$（模拟实际$x_i$）训练模型

#### 验证

- 本地验证(todo)
- 生成vector和meta，导入https://projector.tensorflow.org/ (UMAP)

#### 结果

- 用lr=0.0001训练15 epoch，loss就几乎接近于0
- loss接近于0的情况下，模型结果完全没有区分 T 和 noise,尽管T的聚类结果很好

#### 改进

- [ ] 用随机生成的noise代替随机截图
- [ ] 验证的时候,尽可能多的编码t（中心可点击）
- [ ] 编写metric检测，把随机截图和T放入验证集

### 分块对比法-classification

### 分块对比法-GAN

用GAN的方法，可以同时得到G和D，值得一试

### FCN

SPP/FASTER R-CNN/YOLOV3
这个时我最不想做的，因为需要带坐标的数据
