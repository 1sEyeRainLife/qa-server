from langchain.agents import AgentType, initialize_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from langchain.llms import Ollama

# 初始化工具
search = DuckDuckGoSearchRun()
tools = [Tool(name="Search", func=search.run, 
             description="用于搜索最新信息")]

# 创建Agent
agent = initialize_agent(
    tools=tools,
    llm=Ollama(model="llama3.2", temperature=0),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 执行任务
# agent.run("2023年诺贝尔文学奖得主是谁？")