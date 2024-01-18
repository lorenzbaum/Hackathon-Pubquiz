# import the necessary modules
import os

from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI

from audio_agent import audio_speech_tool
from documents import document_tool
from duckduckgo import ddg_tool
from example_questions import EXAMPLE_QUESTIONS
from math_tool import wolfram_tool
from wikipedia_tool import wikipedia_tool

load_dotenv(override=True)

azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')

# create llm instance

llm = AzureChatOpenAI(
    api_key=azure_api_key,
    api_version="2023-05-15",
    azure_deployment="gpt-35-turbo-16k",
    azure_endpoint=azure_endpoint,
)

# append duck and go
tools = [ddg_tool]

simple_summary_prompt = ChatPromptTemplate.from_template("""Please summarize the following piece of text.
Respond in a manner that a 5 year old would understand.

Text: {input}""")
simple_summary_chain = {"input": RunnablePassthrough()} | simple_summary_prompt | llm
summary_tool = Tool(
    name="Summary Tool",
    func=simple_summary_chain.invoke,
    description="Use this tool to do a summary. Make sure you get the text to do a summary of first."
)
tools.append(summary_tool)

tools.append(document_tool)

tools.append(audio_speech_tool)

tools.append(wikipedia_tool)

# tools.append(tool_llm_math)

tools.append(wolfram_tool)

# append summary

PREFIX = """
You are participating in a pubquiz. Answer in a short sentence.

Hint: Do you have every information you need for answering the question?
"""

memory = ConversationBufferMemory(memory_key="chat_history")

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
    agent_kwargs={
        'prefix': PREFIX
    }
)

if __name__ == "__main__":
    agent.invoke("Welcher Paragraph des deutschen Strafgesetzbuch handelt von Beihilfe?")

    # agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    agent.invoke("How much is 2+2")

    for question in EXAMPLE_QUESTIONS:
        agent.invoke(question)
        print("----------------------------------")
