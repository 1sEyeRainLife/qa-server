from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import RedisChatMessageHistory, ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.chains import ConversationChain


def get_user_memory(sid: str) -> ConversationBufferMemory:
    rkey = f"chat_history:{sid}"
    mhistory = RedisChatMessageHistory(
        url="redis://localhost:6379/0",
        ttl=60,
        session_id=rkey
    )
    return ConversationBufferMemory(
        chat_memory=mhistory,
        memory_key="history",
        output_key="history"
    )

callback = AsyncIteratorCallbackHandler()

ollama_llm = ChatOllama(
    model="qwen3",
    temperature=0.7,
    base_url="http://localhost:11434",
    disable_streaming=False,
    callbacks=[callback]
)


shared_chain = ConversationChain(
    llm=ollama_llm
)

async def chat(sid, question):
    shared_chain.prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的AI助手，请用中文回答"),
        ("human", "{query}"),
        ("ai", "{history}")
    ])
    shared_chain.memory = get_user_memory(sid)
    shared_chain.run(question)

    async for m in callback.aiter():
        yield m