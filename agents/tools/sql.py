import sqlite3
from langchain.tools import Tool

conn = sqlite3.connect("db.sqlite")

"""
   Langchain adds another layer of abstraction and hides the complex details of prompting and ChatGPT code 
   'run_sqlite_query' function is called when ChatGPT responses with a query. The response of this function
   call is sent back to the ChatGPT that gives the final result that can be sent back to he user.
"""


def run_sqlite_query(query):
    c = conn.cursor()
    c.execute(query)
    return c.fetchall()


run_query_tool = Tool.from_function(
    name="run_sqilite_query", description="Run a sqlite query.", func=run_sqlite_query
)
