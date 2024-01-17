from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool


wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

wikipedia_tool = Tool.from_function(
    func = wikipedia.run,
    name = "Wikipedia Search",
    description = "Search wikipedia encyclopedia for common questions.",
)
