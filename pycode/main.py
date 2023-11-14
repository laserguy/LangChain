from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import argparse

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

"""
    OpenAI looks for the variable OPENAI_API_KEY while returning the LLM if not passed explicity as a
    command line parameter. 'load_dotenv' loads the environment variables mention in '.env' file.
    IMP:- '.env' file should not be commited
"""
llm = OpenAI()

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"],
)

code_chain = LLMChain(llm=llm, prompt=code_prompt)

result = code_chain({"language": args.language, "task": args.task})

print(result["text"])
