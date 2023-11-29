######################################################################################################
"""
    This file was created/copied from 'main.py', to separately show how the problem that main.py is addressed
    All the comments of 'main.py' are removed here.
    
    We were getting the issue in 'main.py' because ChatGPT did not have the information about the database.
    This will be addressed here by giving database information through the 'system' message. Check 'tool_6.png'
"""
######################################################################################################

from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from dotenv import load_dotenv

from tools.sql import run_query_tool, list_tables, describe_tables_tool

load_dotenv()

chat = ChatOpenAI()

tables = list_tables()

prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(
            content=f"You are an AI that has access to a SQLite database.\n{tables}"
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Adding multiple tools mean that each tool will go through a iteration of back and forth with the ChatGPT
# Sometimes ChatGPT can decide not to use a tool
# Here in this case adding this tool doesn't solve this issue
tools = [run_query_tool, describe_tables_tool]

agent = OpenAIFunctionsAgent(llm=chat, prompt=prompt, tools=tools)

agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools)

agent_executor("How many users are in the database that have shipping address?")
