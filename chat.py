import asyncio
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import RedisChatMessageHistory, ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler, AsyncCallbackHandler
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.chains import ConversationChain


class CustomAsyncIteratorCallbackHandler(AsyncCallbackHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._queue = asyncio.Queue()
        self._done = asyncio.Event()

    async def on_chat_model_start(self, *args, **kwargs):
        pass

    async def on_llm_new_token(self, token, **kwargs):
        self._queue.put_nowait(token)

    async def on_llm_end(self, *args, **kwargs):
        self._done.set()

    async def aiter(self):
        while not self._queue.empty() or not self._done.is_set():
            if not self._queue.empty():
                yield await self._queue.get()
            else:
                await asyncio.sleep(0.5)


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
    )

callback = CustomAsyncIteratorCallbackHandler()

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
        ("human", "{input}"),
        ("ai", "{history}")
    ])
    shared_chain.memory = get_user_memory(sid)
    shared_chain.run(question)

    async for m in callback.aiter():
        yield m


async def main():
    async for m in chat("fanbin", "李世民有什么功绩？"):
        print(m)

if __name__ == "__main__":
    asyncio.run(main())