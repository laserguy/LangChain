######################################################################################################
"""
    To understand this, it is important to follow along with images(documentation_images):
    What does the using of tool means in ChatGPT: Telling ChatGPT using prompt that it has the access
        to some tool, and it has to create some query and give to us, that can be used to access the tool
        to retrieve the required information.
        Can be understood from the flow.png and tool_1.png files.
        
    tool_2.png: shows the prompt that can be used to use ChatGPT along with the tool:
        Our program will run the query provided by the ChatGPT on sqlite DB and return the response to
        the ChatGPT, and ChatGPT will send it back to user in a more human/natural format.
        
    tool_3.png: ChatGPT has provided functionality to the same thing as shown in tool_2.png in a more
        programmer friendly manner
        
    tool_4.png: Check ./tools/sql.py to see(Adds another layer of abstraction)
"""
######################################################################################################
