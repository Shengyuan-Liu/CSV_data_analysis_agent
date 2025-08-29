# CSV 数据分析 Agent

本项目实现了一个基于 LangChain 的数据分析 Agent。所有数据探索与建模任务均由大模型自动生成并通过 Python REPL 工具执行（不是手写分析代码）。你可以通过自然语言与 Agent 对话，在每次对话中完成一个功能。

数据文件：`data/titanic_cleaned.csv`（约 800+ 行）

主要功能（每次对话完成一个）：
- 数据 summary：对数值列统计均值、方差、最小值、最大值等
- 缺失值填充：数值列用均值填充，并另存为新 CSV
- 画图：例如绘制 `Survived` 列的分布图
- 模型训练与预测：使用 sklearn，输出精度与示例预测，并保存模型

实现方式由大模型完成：
- Agent 使用 LangChain 的 `PythonREPLTool` 工具。大模型会在每次对话中自行产出完整 Python 代码（导入 pandas、matplotlib、sklearn 等），并在 REPL 中执行。
- 我们不手写具体分析/绘图/训练逻辑，所有计算、图像、模型训练与预测都由 LLM 生成代码完成。

## 环境准备

1) Python 版本
- 建议 Python 3.10+

2) 安装依赖
```bash
pip install -r requirements.txt
```

3) 配置环境变量（.env）
在项目根目录创建 `.env`，示例：
```
OPENAI_API_KEY=你的APIKey
BASE_URL=https://api.zhizengzeng.com/v1
OPENAI_MODEL=gpt-4o
```

4) 目录说明
- `data/titanic_cleaned.csv`：固定读取的数据集
- `output/`：图像与模型输出目录（程序会自动创建）

## 运行

```shell
python agent.py
```

启动后会进入一个简单的 CLI 对话界面。请按“每次对话一个功能”的要求，依次输入以下 4 条指令（示例）：

1) 数据 summary
```
请对数据做 summary（均值、方差、最小、最大等）
```

2) 缺失值均值填充
```
请对数值列的缺失值用均值填充，并将结果保存到 data/titanic_cleaned_filled.csv，同时打印填充前后缺失计数
```

3) 绘制 Survived 分布
```
请绘制 Survived 列的分布图，保存到 output/survived_distribution.png
```

4) 训练模型并预测
```
请用 sklearn 训练一个分类模型完成预测，打印 accuracy 和若干条预测示例，并将模型保存到 output/model.joblib
```

执行过程中，Agent 会：
- 自动通过 Python REPL 运行大模型生成的代码
- 读取固定数据路径：`data/titanic_cleaned.csv`
- 将图表保存到 `output/`，新 CSV 保存到 `data/`，模型保存到 `output/`
- 在控制台打印关键结果与保存文件的路径

## 生成物位置（预期）
- 缺失值填充后的 CSV：`data/titanic_cleaned_filled.csv`
- 分布图：`output/survived_distribution.png`
- 训练好模型：`output/model.joblib`

## 常见说明与提示
- 如果数据中不存在 `Survived` 列，Agent 会打印可选列名，并提示你选择替代表达的目标列或绘图列。
- 若遇到依赖缺失，请确认已执行 `pip install -r requirements.txt`。
- 想更换数据文件时，可修改 `agent.py` 中的 `DATA_PATH` 常量，或将目标 CSV 放置为 `data/titanic_cleaned.csv`。
