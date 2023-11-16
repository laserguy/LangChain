from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
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

code_test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test for the following {language} code:\n {code} ",
)

"""
    The 'chains' require two important things 'prompt' and 'LLM'
    The output JSON/dictionary will have all the input variables as key, plus also a 'text' key
    containing the output of the prompt
    The 'text' key can be configured/renamed like below we rename it to 'code'
"""
code_chain = LLMChain(llm=llm, prompt=code_prompt, output_key="code")

code_test_chain = LLMChain(llm=llm, prompt=code_test_prompt, output_key="test")

# Put multiple chain in sequence, the output of the first chain is input to the second
chain = SequentialChain(
    chains=[code_chain, code_test_chain],
    input_variables=["task", "language"],
    output_variables=["test", "code"],
)

result = chain({"language": args.language, "task": args.task})

print(">>>>>>>> GENERATED CODE >>>>>>>>>>")
print(result["code"])
print(">>>>>>>> GENERATED TEST >>>>>>>>>>")
print(result["test"])
