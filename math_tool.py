# WOLFRAM_ALPHA_APPID=VTEH48-APTXEW4QVE

import os

from dotenv import load_dotenv
from langchain.chains import LLMMathChain
from langchain_core.tools import Tool
from langchain.agents import load_tools
from langchain_openai import AzureChatOpenAI


load_dotenv(override=True)

azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')

wolfram_tool = load_tools(["wolfram-alpha"])[0]


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
