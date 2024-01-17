from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

ddg_tool = Tool.from_function(
    func = ddg.run,
    name = "Wikipedia Search",
    description = "Search an encyclopedia for common questions.",
)
