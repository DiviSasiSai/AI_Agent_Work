#Weather Agent
from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import load_tools
from langchain.agents import create_tool_calling_agent,AgentExecutor
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
from typing import Annotated
import sys
import io
from IPython.display import display, Markdown
from contextlib import redirect_stdout
import requests
from datetime import datetime
import os

os.environ["SERPAPI_API_KEY"] =""


llm = ChatOpenAI(temperature=0.0,model="gpt-4o-mini",base_url="https://models.inference.ai.azure.com/",
    api_key="")

load_dotenv()
toolbox = load_tools(tool_names=['serpapi'], llm=llm)


@tool
def get_location_from_ip():
    """Get the geographical location of user"""
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        if 'loc' in data:
            latitude, longitude = data['loc'].split(',')
            data = (
                f"Latitude: {latitude},\n"
                f"Longitude: {longitude},\n"
                f"City: {data.get('city', 'N/A')},\n"
                f"Country: {data.get('country', 'N/A')}"
            )
            return data
        else:
            return "Location could not be determined."
    except Exception as e:
        return f"Error occurred: {e}"

@tool
def get_current_datetime() -> str:
    """Return the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

tools = toolbox+[get_current_datetime,get_location_from_ip]

prompt = ChatPromptTemplate.from_messages([
    ("system","you are a helpfull assistent"),
    MessagesPlaceholder(variable_name="history"),
    ("human","{query}"),
    ("placeholder","{agent_scratchpad}")
])

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_exe = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

chat_map={}
def get_session(session_id:str)->InMemoryChatMessageHistory:
  if session_id in chat_map:
    return chat_map[session_id]
  chat_map[session_id] = InMemoryChatMessageHistory()
  return chat_map[session_id]

agent_with_history = RunnableWithMessageHistory(
    agent_exe,
    get_session_history=get_session,
    input_messages_key="query",
    history_messages_key="history"
)



while 1:
  user= input("user: ").strip()

  if user == "stop":
    break
  f = io.StringIO()
  with redirect_stdout(f):
    output = agent_with_history.invoke({"query":user},config={"session_id":"user_id"})
  print("AI: ",output["output"])

