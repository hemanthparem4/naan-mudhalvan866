from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Create a prompt template
template = """The following is a friendly conversation between a human and an AI. 
The AI is talkative and provides lots of specific details from its context. 
If the AI doesn't know the answer to a question, it truthfully says it doesn't know.

Current conversation:
{history}
Human: {input}
AI:"""

prompt = PromptTemplate(
    input_variables=["history", "input"], 
    template=template
)

# Initialize the conversation chain
memory = ConversationBufferMemory()
llm = OpenAI(temperature=0.7)  # Adjust temperature for creativity
conversation = ConversationChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True  # Set to False in production
)

# Chat interface
print("AI Chatbot: Hello! I'm your AI assistant. How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print("AI Chatbot: Goodbye! Have a great day!")
        break
    response = conversation.predict(input=user_input)
    print(f"AI Chatbot: {response}")