import os
from typing import Annotated
from openai import AsyncOpenAI

from dotenv import load_dotenv

from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function
import requests

import random
from datetime import datetime
import webbrowser

import asyncio



# Define a sample plugin for the sample

class TimePlugin:
    """To printig time."""
    @kernel_function(description="Provides time to user.")
    def get_time(self) -> Annotated[str, "Returns a time."]:
        # Get current date and time

        now = datetime.now()

        # Format the time
        current_time = now.strftime("%H:%M:%S")
        
        return current_time
    @kernel_function(description="to get weather report")
    def get_weather(self,city):
        base_url = ""
        api_key=""
        city="Bapatla"
        params = {
            "key": api_key,
            "q": city,
            # you can also include optional params like "aqi": "yes"
        }

        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            # data["location"] has info about the place
            # data["current"] has weather info
            loc = data.get("location", {})
            curr = data.get("current", {})

            name = loc.get("name")
            region = loc.get("region")
            country = loc.get("country")
            temp_c = curr.get("temp_c")
            condition = curr.get("condition", {}).get("text")
            humidity = curr.get("humidity")
            wind_kph = curr.get("wind_kph")

            return f"Weather in {name}, {region}, {country} Condition: {condition} Temperature: {temp_c} °C Humidity: {humidity}% Wind Speed: {wind_kph} kph"
        else:
            return f"Failed to fetch weather. Status code: {response.status_code}"
    @kernel_function(description="to open the youtube")
    def open_youtube(self):
      # URL of YouTube
      url = "https://www.youtube.com"

      # Open YouTube in the default browser
      webbrowser.open(url)



load_dotenv()
client = AsyncOpenAI(
    api_key= "",
    base_url="",
)

# Create an AI Service that will be used by the `ChatCompletionAgent`
chat_completion_service = OpenAIChatCompletion(
    ai_model_id="gpt-4o",
    async_client=client,
)


agent = ChatCompletionAgent(
    service=chat_completion_service,
    name="ChatAgent",
    plugins=[TimePlugin()],
    instructions="You are a helpful AI Agent that can help chat with user",
)

async def main():
    # Create a new thread for the agent
    # If no thread is provided, a new thread will be
    # created and returned with the initial response
    thread: ChatHistoryAgentThread | None = None

    while True:

      user_input = input("user:")

      if user_input == "stop":
        break

      first_chunk = True
      async for response in agent.invoke_stream(
          messages=user_input, thread=thread,
      ):
          # 5. Print the response
          if first_chunk:
              print(f"# {response.name}: ", end="", flush=True)
              first_chunk = False
          print(f"{response}", end="", flush=True)
          thread = response.thread

      print()


    # Clean up the thread
    await thread.delete() if thread else None


asyncio.run(main())










      
