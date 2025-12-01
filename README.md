# LightAD - 轻量级日志异常检测工具

 **基于经典机器学习的高性能日志异常检测** | **MCP 服务即插即用**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastMCP](https://img.shields.io/badge/FastMCP-2.0%2B-green)](https://github.com/jlowin/fastmcp)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Paper](https://img.shields.io/badge/ICSE-2024-red)](https://github.com/BoxiYu/LightAD)

## 原作者与MCP封装作者联系邮箱

- 原项目: https://github.com/BoxiYu/LightAD
- Issues: https://github.com/BoxiYu/LightAD/issues
- 工具制作作者Email: boxiyu@link.cuhk.edu.cn
- MCP项目：https://github.com/Plc912/LightAD-master.git
- MCP封装作者Email:3522236586@qq.com

基于 ICSE'24 论文 "Deep Learning or Classical Machine Learning? An Empirical Study on Log-Based Anomaly Detection" 实现，在主流日志数据集上达到 **SOTA 性能**。

<img src=table2.png style="width:50%;height:auto;">

---

## ✨ 特性亮点

- ✅ **轻量高效**：经典机器学习算法，训练速度快 10-50 倍
- ✅ **SOTA 性能**：在 HDFS/BGL 等数据集上达到最先进水平
- ✅ **MCP 服务**：支持 Cursor/Claude 无缝集成
- ✅ **异步任务**：后台运行，支持长时间训练
- ✅ **自动优化**：贝叶斯优化自动选择最优超参数
- ✅ **多数据集**：支持 HDFS 和超级计算机日志

---

## 📦 快速开始

### 1. 安装依赖

```bash
# 克隆或进入项目目录
cd LightAD-main

# 安装依赖（注意解决 torch 版本冲突）
pip install packaging  # 先安装 packaging
pip install -r requirements.txt
```

### 2. 启动 MCP 服务

```bash
# 方式一：直接启动
python lightad_mcp_server.py

# 方式二：使用脚本
./start_mcp_server.sh    # Linux/Mac
start_mcp_server.bat     # Windows
```

服务默认在 **http://127.0.0.1:2224** 启动

### 3. 配置客户端

**Cherry Studio 配置**：

```json
{
  "mcpServers": {
     "E3P3NoGSxSm0W42t6N1BP": {
      "name": "lightad-master",
      "type": "sse",
      "description": "基于经典机器学习的高性能日志异常检测",
      "isActive": true,
      "baseUrl": "http://127.0.0.1:2224/sse",
      "installSource": "unknown"
    }
  }
}
```

### 4. 开始使用

#### 使用 Python API

```python
# 训练 KNN 模型（使用示例数据集）
result = lightad_train_hdfs(model="knn", eliminate=False)
task_id = result["task_id"]

# 查询训练状态
status = get_task(task_id)
print(f"进度: {status['progress']:.1%}")

# 获取结果
if status["status"] == "succeeded":
    metrics = status["result"]["average_results"]
    print(f"F1-Score: {metrics['f1_score']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
```

---

## 🛠️ 可用工具

| 工具                       | 功能                  | 数据集                   |
| -------------------------- | --------------------- | ------------------------ |
| `lightad_train_hdfs`     | 训练 KNN/DT/SLFN 模型 | HDFS                     |
| `lightad_train_super`    | 语义匹配异常检测      | BGL/Spirit/TBird/Liberty |
| `lightad_optimize_model` | 贝叶斯优化超参数      | HDFS (deduplicated)      |
| `lightad_preprocess`     | 预处理原始日志        | All                      |
| `list_tasks`             | 列出所有任务          | -                        |
| `get_task`               | 查询任务详情          | -                        |

---

## 📊 性能表现

### HDFS 数据集（100K 样本）

| 模型           | F1-Score | 训练时间 | 推理时间 |
| -------------- | -------- | -------- | -------- |
| **KNN**  | 0.93+    | 2-5s     | 0.3s     |
| **DT**   | 0.91+    | 1-3s     | 0.1s     |
| **SLFN** | 0.90+    | 10-20s   | 0.5s     |

### 与深度学习对比

- ⚡ **训练速度**：快 10-50 倍
- 💾 **内存占用**：少 5-10 倍
- 🎯 **准确率**：相当或更高
- 🔧 **调参难度**：自动优化

---

### 数据准备

#### HDFS 数据集

```python
# 方式 1：使用示例数据（已包含 100K 样本）
lightad_train_hdfs(model="knn")

# 方式 2：准备完整数据
# 下载：https://doi.org/10.5281/zenodo.1144100
# 放到：datasets/original_datasets/hdfs
lightad_preprocess(dataset="hdfs", eliminate=False)
lightad_train_hdfs(model="knn")
```

#### 超级计算机数据集

```python
# 下载数据：https://www.usenix.org/cfdr-data
# 放到：datasets/original_datasets/{dataset_name}
lightad_preprocess(dataset="bgl")  # 或 spirit/tbird/liberty
lightad_train_super(dataset="bgl", sample_ratio=0.1)
```

### 使用场景

#### 场景 1：快速验证

```
"帮我用 KNN 模型训练 HDFS 数据集"
"使用决策树模型训练 HDFS 数据"
"训练一个神经网络模型用于日志异常检测"
"用 SLFN 模型训练，隐藏层大小设为 50"
```

**Python API**：

```python
# 使用示例数据，快速体验
result = lightad_train_hdfs(model="knn")
```

#### 场景 2：模型优化

```
"自动优化 KNN 模型的超参数，我更看重准确率"
"帮我找到最优的决策树参数，准确率权重 0.7，训练时间和推理时间各 0.15"
"使用贝叶斯优化为 SLFN 模型选择最佳配置"
"优化 KNN 模型，准确率和推理速度各占 50% 权重"
```

**Python API**：

```python
# 自动优化模型超参数
result = lightad_optimize_model(
    model="knn",
    l1=0.7,  # 70% 权重给准确率
    l2=0.15, # 15% 权重给训练时间
    l3=0.15  # 15% 权重给推理时间
)
```

#### 场景 3：多模型对比

```
"对比 KNN、决策树和神经网络三个模型的性能"
"分别用 KNN、DT 和 SLFN 训练，然后告诉我哪个最好"
"帮我测试所有可用模型的表现"
```

**Python API**：

```python
models = ["knn", "dt", "slfn"]
for model in models:
    result = lightad_train_hdfs(model=model)
    # 对比 F1-Score...
```

#### 场景 4：任务管理

```
"查看所有训练任务的状态"
"显示任务列表"
"检查我的训练任务完成了没有"
"获取任务 [task_id] 的详细结果"
"显示最新完成任务的 F1 分数"
```

**Python API**：

```python
# 列出所有任务
tasks = list_tasks()

# 查询特定任务
status = get_task(task_id)
```

#### 场景 5：数据预处理

```
"预处理 HDFS 数据集，不使用去重"
"帮我预处理 BGL 数据集"
"预处理 HDFS 数据并启用去重功能"
"使用 80% 的数据作为训练集预处理 HDFS"
```

**Python API**：

```python
# 预处理数据
result = lightad_preprocess(dataset="hdfs", eliminate=False)
```

---

## 📖 命令行使用

如果不使用 MCP 服务，也可以直接使用命令行：

### 预处理数据集

```bash
python preprocess.py --dataset hdfs
```

### 训练模型

```bash
# HDFS 数据集
python main_hdfs.py --model knn

# 去重数据集
python main_hdfs.py --model knn --eliminate True

# 超级计算机数据集
python main_super.py --dataset bgl
```

### 模型优化

```bash
python main_opt.py --model knn --l1 0.7 --l2 0.15 --l3 0.15
```

---

## 🔧 高级配置

### 环境变量

```bash
# 设置最大并发任务数
export LIGHTAD_MAX_CONCURRENT=4
```

### 远程部署

```bash
# 1. 服务器启动（开放 2224 端口）
python lightad_mcp_server.py

# 2. 客户端配置
{
  "url": "http://your-server-ip:2224/sse"
}
```

---

## 💬 自然语言参考

### 基础训练

```
"帮我用 KNN 模型训练 HDFS 数据集"
"使用决策树模型训练 HDFS 数据"
"训练一个单层神经网络模型"
"用 KNN 训练，邻居数设为 3"
"在去重的 HDFS 数据上训练 KNN 模型"
```

### 模型对比

```
"对比 KNN 和决策树的性能"
"分别用 KNN、DT 和 SLFN 训练，然后告诉我哪个最好"
"测试所有可用模型并给出推荐"
"比较一下各个模型的训练时间和准确率"
```

### 超参数优化

```
"自动优化 KNN 模型，我更看重准确率"
"帮我找到最优的决策树参数"
"优化模型参数，准确率权重 0.7，训练时间 0.15，推理时间 0.15"
"使用贝叶斯优化找到最佳配置"
```

### 数据预处理

```
"预处理 HDFS 数据集"
"预处理数据，不使用去重"
"帮我预处理 BGL 数据集"
"预处理 HDFS 数据并启用去重功能"
```

### 任务管理

```
"查看所有训练任务"
"显示任务列表"
"检查训练任务完成了没有"
"查询任务 abc-123-def 的状态"
"获取最新任务的结果"
"显示任务进度"
```

### 结果查询

```
"显示最新完成任务的 F1 分数和准确率"
"查看模型的训练时间和推理时间"
"告诉我模型优化的最佳参数"
"对比各个模型的性能指标"
```

### 超级计算机数据集

```
"在 BGL 数据集上进行异常检测"
"训练 Spirit 数据集，采样率 0.1"
"使用语义匹配方法分析 TBird 日志"
"预处理 Liberty 数据集"
```

### 完整工作流

```
"先预处理 HDFS 数据，然后用 KNN 训练"
"预处理完成后，对比 KNN 和决策树的性能"
"优化 KNN 模型，然后用最优参数重新训练"
"帮我完成从数据预处理到模型训练的全流程"
```

记得询问的时候添加上数据地址如：E:\\......\lightad-master\LightAD-main\datasets\original_datasets\hdfs。

---

## 📖 引用

基于 ICSE'24 论文实现：

```bibtex
@inproceedings{lightad2024,
  title={Deep Learning or Classical Machine Learning? An Empirical Study on Log-Based Anomaly Detection},
  author={Yu, Boxi and others},
  booktitle={ICSE},
  year={2024}
}
```

---

## 🤝反馈

如有问题，请：

- 提交 [Issue](https://github.com/BoxiYu/LightAD/issues)
- 原项目作者邮件联系: boxiyu@link.cuhk.edu.cn
- MCP封装作者联系:3522236586@qq.com
