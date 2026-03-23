import os
import requests

from langchain_openrouter import ChatOpenRouter
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage


SYSTEM_PROMPT = """
You are an searchin assistant.
Use search_with_web if you need get information from web.
"""


@tool
def search_with_web(url: str) -> str:
    """Get text from URL."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


llm = ChatOpenRouter(
    model="stepfun/step-3.5-flash:free",
    openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0.1,
    max_retries=2,
)

agent = create_agent(
    model=llm, tools=[search_with_web],
)

messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content="Get summarize of  https://langchain-ai.github.io/langgraph/llms.txt. What this page is about?"),
]


res_with_tool = agent.invoke({"messages": messages})
simple_result = agent.invoke({"messages": [HumanMessage(content="What is 2 + 2")]})
