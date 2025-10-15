#Simple chat agent
#tools are get location and get date and time

from langchain.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent,AgentExecutor
from langchain.tools import tool
from langchain_core.runnables.history import RunnableWithMessageHistory

from datetime import datetime
import requests

@tool
def get_date_and_time():
  """get the current date and time"""
  return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
@tool
def get_location():
  """get the current location of user"""
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
    
llm = ChatOpenAI(temperature=0.0,model="gpt-4o-mini",base_url="https://models.inference.ai.azure.com/",
    api_key="")

prompt = ChatPromptTemplate.from_messages([
 ("system","you are simple chat bot"),
 MessagesPlaceholder(variable_name="history"),
 ("human","{query}"),
 ("placeholder","{agent_scratchpad}") 
])

tools = [get_date_and_time,get_location]

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_exe = AgentExecutor(
    agent = agent,
    tools = tools,
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

import sys
import io
from contextlib import redirect_stdout


while 1:
  user= input("user: ").strip()

  if user == "stop":
    break
  f = io.StringIO()
  with redirect_stdout(f):
    output = agent_with_history.invoke({"query":user},config={"session_id":"user_id"})
  print("AI: ",output["output"])
