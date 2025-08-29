import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage

# Load environment variables
load_dotenv()

# Configure LLM (OpenAI-compatible, e.g., DeepSeek via BASEURL)
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "deepseek-chat"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("BASEURL"),
    temperature=0.2,
)

# Single tool: Python REPL for executing LLM-generated Python code
python_tool = PythonREPLTool()
tools = [python_tool]

# Fixed dataset path (detected from repository)
DATA_PATH = Path("data/titanic_cleaned.csv").as_posix()

SYSTEM_PROMPT = """你是一个资深 Python 数据分析 Agent。你只能通过一个工具 python_repl 来执行任意 Python 代码，从而完成所有数据分析、可视化与建模任务。
严禁在脑中计算或伪造结果，所有统计、图表与模型训练/预测都必须通过 python_repl 工具的代码来完成与输出。

数据集路径（请在代码中使用它读取数据）:
{data_path}

约束与要求：
- 任何数据处理/分析/建模与可视化均需通过 python_repl 工具执行的 Python 代码完成，不要在思维中直接给出计算结果。
- Python 代码中总是显式导入所需库，例如：pandas、numpy、matplotlib/seaborn、scikit-learn、joblib 等。
- 所有文件输出（图片/模型/新CSV）统一保存到当前项目的相对路径下：
  - 图片：output/ 目录（如 output/survived_distribution.png）
  - 填充后的CSV：data/ 目录（如 data/titanic_cleaned_filled.csv）
  - 训练好的模型：output/ 目录（如 output/model.joblib）
  代码应在保存前确保目录存在（若不存在请在代码中创建）。
- 执行完成后，代码需要使用 print() 输出关键结果与所有生成文件的相对路径，便于用户查找。
- 当列名缺失或与预期不符（例如没有 'Survived' 列）时，代码应打印可选列名并给出下一步建议，而不是报错中断。

四类典型任务的参考实现要点（供你在代码中遵循，不要在脑中直接回答结果）：
1) 数据 summary：
   - 读取 {data_path} 为 df
   - 对数值列输出均值、方差、最小值、最大值；可以结合 df.describe(include='all')
   - 清晰打印结果（例如以表格或分块打印）

2) 缺失值均值填充：
   - 仅对数值列做均值填充，不改变非数值列类型
   - 保存到 data/titanic_cleaned_filled.csv
   - 打印填充前后缺失计数变化以及输出文件路径

3) 绘制 Survived 列分布图：
   - 若列存在：绘制直方图/柱状图（任选其一），设置标题与标签
   - 确保 output/ 目录存在并保存到 output/survived_distribution.png
   - 打印图片保存路径
   - 若列不存在：打印 df.columns 供用户选择，并提示可替代列名

4) 使用 sklearn 训练模型并预测：
   - 优先以 'Survived' 作为目标列（若存在），否则打印列名并等待用户指定目标
   - 做必要的数值预处理（仅示例化处理，避免过度复杂）
   - 训练简单可解释的模型（如 LogisticRegression 或 RandomForestClassifier）
   - 打印 accuracy、classification_report，并展示若干条预测示例
   - 使用 joblib.dump 保存模型到 output/model.joblib，并打印该路径

与工具交互协议：
- 在计划好步骤后，你应调用 python_repl 工具，并将完整、可直接运行的 Python 源码作为唯一的 'code' 实参传入。
- 不要在工具参数里使用 Markdown 代码块标记（不要使用 ``` 包裹），仅提供纯 Python 代码字符串。
- 代码必须自包含：导入依赖、读取 CSV（使用固定路径 {data_path}）、执行任务、打印关键信息与输出文件路径。

返回给用户的信息格式：
- 首先简要说明你的思考与计划（不含任何脑算结果）
- 然后调用工具执行代码
- 工具执行后，总结关键结果与生成文件位置（引用已由代码打印的路径）
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


def run_cli():
    print("CSV Data Agent (LangChain + python_repl)")
    print(f"- 固定数据路径: {DATA_PATH}")
    print("- 输出目录: output/  (图表与模型将保存在此)")
    print("- 示例对话：")
    print("  1) 请对数据做 summary（均值、方差、最小、最大等）")
    print("  2) 请对数值列的缺失值用均值填充，并输出到 data/titanic_cleaned_filled.csv")
    print("  3) 请绘制 Survived 列的分布图，保存到 output/survived_distribution.png")
    print("  4) 请用 sklearn 训练一个模型完成预测，并给出精度与若干预测结果示例")
    print("输入 'exit' 退出。\n")

    chat_history = []

    # Ensure output directory exists (non-destructive helper)
    Path("output").mkdir(parents=True, exist_ok=True)

    while True:
        try:
            user_input = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("再见。")
            break

        result = agent_executor.invoke(
            {
                "input": user_input,
                "chat_history": chat_history,
                "data_path": DATA_PATH,
            }
        )

        # Print assistant output
        output_text = result.get("output", "")
        print("\n助手:", output_text, "\n")

        # Maintain simple chat history for better multi-turn performance
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=output_text))


if __name__ == "__main__":
    run_cli()
