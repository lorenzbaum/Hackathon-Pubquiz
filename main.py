# Load OpenAI key from env

import os

from dotenv import load_dotenv

load_dotenv(override=True)

azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')

# create llm instance


# import the necessary modules
from audio_agent import audio_speech_tool
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.agents import initialize_agent, AgentType
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain.tools.ddg_search import DuckDuckGoSearchRun
from langchain import LLMMathChain
from documents import document_tool
from wikipedia_tool import wikipedia_tool

embeddings = AzureOpenAIEmbeddings(
    api_key=azure_api_key,
    api_version="2023-05-15",
    azure_deployment="text-embedding-ada-002",
    azure_endpoint=azure_endpoint,
)

llm = AzureChatOpenAI(
    api_key=azure_api_key,
    api_version="2023-05-15",
    azure_deployment="gpt-35-turbo-16k",
    azure_endpoint=azure_endpoint,
)

llm_math = LLMMathChain.from_llm(llm, verbose=True)

tool_llm_math = Tool.from_function(
    func=llm_math.run,
    name="calculate mathematical questions",
    description="Use math for questions",
)

ddg = DuckDuckGoSearchRun()

ddg_tool = Tool.from_function(
    func=ddg.run,
    name="DuckDuckGo Search",
    description="Search DuckDuckGo for a query abount current events.",
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

tools.append(tool_llm_math)

# tools.append(math_tool)


# tools.append(wikipedia)

# append summary

PREFIX = """
You are participating in a pubquiz. Answer in a short sentence.

Hint: Do you have every information you need for answering the question?
"""

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    agent_kwargs={
        'prefix': PREFIX
    }
)

# a = agent.run("Welcher Paragraph des deutschen Strafgesetzbuch handelt von Beihilfe?")

# print(a)


# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
#agent.invoke({"input": "How much is 2+2"})
