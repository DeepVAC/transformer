# transformer
**符合DeepVAC规范的transformer实现。**  
DeepVAC-compliant transformer implementation.

# 运行
### MLab HomePod 2.0
在MLab HomePod 2.0上运行该项目，无需安装任何依赖，只需要输入如下命令即可开始训练:
```bash
python train.py
```

### 自定义环境
需要至少安装：
- pytorch 1.8+;
- numpy
- deepvac 0.6.0+

然后输入如下命令即可开始训练：
```bash
python train.py
```

# 配置
该项目符合DeepVAC规范，相关配置均包含在config.py中，配置项的含义也符合DeepVAC规范，请自行阅读[DeepVAC规范](https://github.com/deepvac/deepvac)。
