from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

"""
    There are two kind of LLMs:
        1 - Completion type
        2 - Conversational type (single turn/multi turn)
        
    There are three important labels that OpenAI uses to distinguish the messages:
        - System => How chat system was configured, this could be the inital setup to describe the
            personality of the chatbot, like "You are a programmer, who helps people in writing programs"
            In the OpenAI this message will stay like this label:
            System: <Message>     (Just first time)
        - User/Human => messages from human
            User: <Message>, Langchain uses 'Human' => Human: <Message> 
        - Assistant => response from the AI
            Assistant: <Message>
            
            
    For chatting it is important to maintain the history of conversation
"""

chat = ChatOpenAI()

prompt = ChatPromptTemplate(
    input_variables=["content"],
    messages=[HumanMessagePromptTemplate.from_template("{content}")],
)

chain = LLMChain(llm=chat, prompt=prompt)

while True:
    content = input(">>")
    result = chain({"content": content})

    print(result["text"])
