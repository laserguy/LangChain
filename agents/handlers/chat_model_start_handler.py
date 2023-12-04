################################################################################
"""
    It's difficult to debug in Langchain, using 'verbose=True' does help.
    But how do we log information that might be helpful in debugging issues.
    Internally many messages can be sent to the to the LLM, how to know them, log them etc.
    
    That is where 'handlers' come in picture, they have registered some callbacks that will be called
    when the respective event occurs. For more detials of events see 'handler.png'
    
    For more understanding see video 61
"""
################################################################################

from langchain.callbacks.base import BaseCallbackHandler
from pyboxen import boxen


def boxen_print(*args, **kwargs):
    print(boxen(*args, **kwargs))


class ChatModelStartHandler(BaseCallbackHandler):
    # Below callback is a special handler which will be called when 'on_chat_model_start' event occurs
    def on_chat_model_start(self, serialized, messages, **kwargs):
        print("\n\n=============Sending Messages============\n\n")

        for message in messages[0]:
            if message.type == "system":
                boxen_print(message.content, title=message.type, color="yellow")

            elif message.type == "human":
                boxen_print(message.content, title=message.type, color="green")

            # message.type where ChatGPT is requesting a function call is still treated as AI message,
            # Therefore, it has to be chceked separately
            elif message.type == "ai" and "function_call" in message.additional_kwargs:
                call = message.additional_kwargs["function_call"]
                boxen_print(
                    f"Running toll {call['name']} with args {call['arguments']}",
                    title=message.type,
                    color="cyan",
                )

            elif message.type == "ai":
                boxen_print(message.content, title=message.type, color="blue")

            # Respose of the function call after using tool is of type "function"
            elif message.type == "function":
                boxen_print(message.content, title=message.type, color="purple")

            else:
                boxen_print(message.content, title=message.type)
