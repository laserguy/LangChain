from langchain.prompts import (
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import (
    ConversationBufferMemory,
)  # To store conversation messages in the buffer
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

"""
    Memory => Is an important class in Langchain, that can be associated with a chain
    to hold information in that chain, the chain will use that memory whenever the chain is used
    
    ConversationBufferMemory(memory_key="messages", return_messages=True)
        "memory_key"=> All the previous and current messages will be stored here
        "return_messages=True"=> This means that messages are stored in the form of
                Human: <message>
                Assistant: <message>
"""
memory = ConversationBufferMemory(memory_key="messages", return_messages=True)

prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    # current message(content) and old messages combined
    messages=[
        MessagesPlaceholder(
            variable_name="messages"
        ),  # IMP to pass the conversation history
        HumanMessagePromptTemplate.from_template("{content}"),
    ],
)

"""
    Once the memory is associated with the chain
    The 'content' and 'response' are automatically appended to the "messages"
"""
chain = LLMChain(llm=chat, prompt=prompt, memory=memory)

while True:
    content = input(">>")
    result = chain(
        {"content": content}
    )  # The result is automatically appended to the memory in chain

    print(result["text"])
