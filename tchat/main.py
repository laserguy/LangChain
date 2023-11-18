from langchain.prompts import (
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import (
    ConversationSummaryMemory,
    ConversationBufferMemory,
    FileChatMessageHistory,
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

chat = ChatOpenAI(
    verbose=True
)  # verbose flag shows what is happening internally, small detail

"""
    Memory => Is an important class in Langchain, that can be associated with a chain
    to hold information in that chain, the chain will use that memory whenever the chain is used
    
    ConversationBufferMemory(memory_key="messages", return_messages=True)
        "chat_memory" => Used to store the messages into some sort of permanent storage, so that it can be
            reloaded again, when program is restarted.
        "memory_key"=> All the previous and current messages will be stored here
        "return_messages=True"=> This means that messages are stored in the form of
                Human: <message>
                Assistant: <message>
"""
# memory = ConversationBufferMemory(
#     chat_memory=FileChatMessageHistory("messages.json"),
#     memory_key="messages",
#     return_messages=True,
# )
"""
    ConversationSummaryMemory 
        => Purpose: There is a limit, how many messages we can store in the history (LLM limit, ChatGPT limits
            and it also cost more money)
            Therefore, this helps build the summary of the conversation history, and instead of passing the
            whole history, only summary is passed as 'system' message
        
        This makes the process bit slower, ConversationSummaryMemory is separate 'chain' that also uses prompt
        and LLM to create a summary from previous summary and current user 'content'
        And this new summary is passed to another chain, which slows down the process.
"""
memory = ConversationSummaryMemory(
    memory_key="messages", return_messages=True, llm=chat
)

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
chain = LLMChain(llm=chat, prompt=prompt, memory=memory, verbose=True)

while True:
    content = input(">>")
    result = chain(
        {"content": content}
    )  # The result is automatically appended to the memory in chain

    print(result["text"])
