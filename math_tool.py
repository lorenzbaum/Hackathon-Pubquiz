from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper

# WOLFRAM_ALPHA_APPID=VTEH48-APTXEW4QVE

from langchain.agents import load_tools
math_tool = load_tools(["wolfram-alpha"])
