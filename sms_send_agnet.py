from twilio.rest import Client

import os
from typing import Annotated
from openai import AsyncOpenAI

from dotenv import load_dotenv

from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function

import random
from datetime import datetime
import webbrowser

import asyncio



# Define a sample plugin for the sample

class SentSMSPlugin:
    """To sent SMS to target number."""
    @kernel_function(description="To Massage to user given number.")
    def sent_SMS(self, to_number: str, message: str):
        # Twilio credentials (get these from your Twilio account)
        from_number = "+15802587968"
        account_sid = "********"
        auth_token = "********"

        client = Client(account_sid, auth_token)

        try:
            message = client.messages.create(
                body=message,
                from_=from_number,
                to=to_number
                )
        except Exception as e:
            print(f"Error sending message: {e}")


load_dotenv()
client = AsyncOpenAI(
    api_key= "********",
    base_url="********",
)

# Create an AI Service that will be used by the `ChatCompletionAgent`
chat_completion_service = OpenAIChatCompletion(
    ai_model_id="gpt-4o",
    async_client=client,
)

load_dotenv()
client = AsyncOpenAI(
    api_key= "********",
    base_url="https://models.inference.ai.azure.com/",
)


# Create an AI Service that will be used by the `ChatCompletionAgent`
chat_completion_service = OpenAIChatCompletion(
    ai_model_id="gpt-4o",
    async_client=client,
)

agent = ChatCompletionAgent(
    service=chat_completion_service,
    name="ChatAgent",
    plugins=[SentSMSPlugin()],
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










      
