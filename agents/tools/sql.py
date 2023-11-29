import sqlite3
from langchain.tools import Tool

conn = sqlite3.connect("db.sqlite")

"""
   Langchain adds another layer of abstraction and hides the complex details of prompting and ChatGPT code 
   'run_sqlite_query' function is called when ChatGPT responses with a query. The response of this function
   call is sent back to the ChatGPT that gives the final result that can be sent back to he user.
"""


# tool A
def run_sqlite_query(query):
    c = conn.cursor()
    try:
        c.execute(query)
        return c.fetchall()
    except sqlite3.OperationalError as err:
        return f"The following error occurred: {str(err)}"


# But what if the query execution fails, 'run_sqlite_query' handles that as well, check 'tool_5.png' (For tool A)
run_query_tool = Tool.from_function(
    name="run_sqilite_query", description="Run a sqlite query.", func=run_sqlite_query
)


# tool B: Added for 'main_enhanced.py'
def describe_tables(table_names):
    c = conn.cursor()
    tables = ", ".join("'" + table + "'" for table in table_names)
    rows = c.execute(
        f"SELECT sql FROM sqlite_master WHERE type='table' and name IN ({tables})"
    )
    return "\n".join(row[0] for row in rows if row[0] is not None)


def list_tables():
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = c.fetchall()
    return "\n".join(row[0] for row in rows if row[0] is not None)


describe_tables_tool = Tool.from_function(
    name="describe_tables",
    description="Given a list of table names, returns the schema of those tables",
    func=describe_tables,
)
