# handpose_and_yolo
对之前手势物体识别项目进行整理，【部分代码也加了注释】其他的项目后面会在慢慢整理，可以关注一波。

---------------------------------------------------------------------------------------------------------------------------------------------------

更新记录：

​					√ **2022.3.07：手势物体识别功能**

​					√ **2023.12.05：新增手部关键点训练代码**

​					√ **2023.12.07：新增训练部分tensorbard绘图**



***



# 手势物体识别功能使用

详细的使用说明参考：https://blog.csdn.net/z240626191s/article/details/123289979

# 手部关键点训练

训练代码在train.py中。

可采用提供的预权重进行fine tune训练。

输入以下命令开始训练：

```
python train.py --model resnet_50 --train_path [数据集路径] --fintune_model 【fine tune模型路径】--batch_size 16
```

如果是fine tune训练，建议初始学习率(init_lr)设置为5e-4，否则建议设置为1e-3。

损失函数此次采用的是MSE，还可支持wing loss。

训练好的权重会保存在model_exp中，对应的tensorboard会存储在logs中【此处用的是均方误差做的评价指标】



# 权重链接

权重百度云：

链接：https://pan.baidu.com/s/1j0RdWoy75nk2aWNjlHgzhQ 
提取码：yypn

