# from agno.agent import Agent
# from agno.models.google import Gemini
# import os
# from pathlib import Path
# from agno.tools.duckdb import DuckDbTools
# import duckdb


# # Create agent
# agent = Agent(
#     model=Gemini(id="gemini-2.0-flash", api_key=os.environ.get("GEMINI_API_KEY")),
#     description="You are an Analyst specialist in analyzing shopping behavior, creating segments and defining marketing strategies.",
#     instructions=[
#         "Use this file for the data: churned_customers.csv",
#         "This knowledge base is a dataset with customers churned or about to churn our brand. Your job is analyzing their shopping behavior",
#         "Check what are the most frequent products they buy, what they buy together, date since from their last purchase compared to 2011-12-31"
#     ],
#     tools=[DuckDbTools()],
#     show_tool_calls=True,
#     markdown=True,
# )

# # Test
# if __name__ == "__main__":
    
#     # Prompt
#     prompt = "What is the most shopped product in the 'churned_customers.csv'?"

#     # Run agent
#     agent.print_response( prompt, stream=True )


import duckdb

sql = """
with cte as(
    select CustomerID, Description, AVG(Quantity) as avg_qty, DENSE_RANK() OVER(PARTITION BY CustomerID ORDER BY avg_qty DESC) as rank
    from 'churned_customers.csv'
    group by 1, 2
)

select *
from cte
WHERE rank IN (1,2)
"""

print(duckdb.query(sql))