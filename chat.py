from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser


ollama_llm = ChatOllama(
    model="qwen3",
    temperature=0.7,
    base_url="http://localhost:11434"
)

def build_chat_chain(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的AI助手，请用中文回答"),
        ("human", "{input}"),
        ("ai", "{history}")
    ])
    return prompt | llm | StrOutputParser()

ollama_chain = build_chat_chain(ollama_llm)


history = []
def chat(user_input):
    response = ollama_chain.invoke({
        "input": user_input,
        "history": history
    })

    history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=response)
    ])

    return response