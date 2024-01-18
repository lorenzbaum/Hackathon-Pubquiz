from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_core.tools import Tool

ddg = DuckDuckGoSearchRun()

ddg_tool = Tool.from_function(
    func=ddg.run,
    name="DuckDuckGo Search",
    description="Search DuckDuckGo for a query about current events.",  # TODO can be optimized
)
