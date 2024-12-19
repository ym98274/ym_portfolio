from phi.agent import Agent
#from phi.model.groq import Groq
from phi.model.openai import OpenAIChat
from dotenv import load_dotenv
from phi.tools.duckduckgo import DuckDuckGo

load_dotenv()


# web_agent_groq = Agent(
#     name="Web Agent",
#     model= Groq(id="llama-3.3-70b-versatile"),
#     tools=[DuckDuckGo()],
#     instructions=["Only return the response to the user query and use available tools if necessary"],
#     show_tool_calls=True,
#     markdown=True,
# )


web_agent_openai = Agent(
    name="Web Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True,
)



#web_agent_groq.print_response("Who won the 2024 constructors championship? and how many points did they win it by?")