import os
from phi.agent import Agent
from phi.model.openai import OpenAIChat
from phi.tools.duckduckgo import DuckDuckGo

# Get the OpenAI API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

# Define the agent using the OpenAI model
web_agent_openai = Agent(
    name="Web Agent",
    model=OpenAIChat(id="gpt-4o", api_key=openai_api_key),  # Pass the API key here
    tools=[DuckDuckGo()],
    instructions=["Only return the response to the user query and use available tools if necessary"],
    show_tool_calls=True,
    markdown=True,
)
