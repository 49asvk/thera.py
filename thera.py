import os
import gradio as gr

#Use your own openai type, version, base and key.
os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-09-15-preview"
os.environ["OPENAI_API_BASE"] = "https://express.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory

llm = AzureOpenAI(
    deployment_name="langchain",
    model_name="gpt-35-turbo-instruct",
)

memory = ConversationBufferMemory()
template = """You are a {adjective} health therapist created by Calyx./
You engage in empathetic and non-judgemental conversations with users, offering coping strategies, relaxation techniques, and helpful advice./
You ALWAYS offer solutions when a user tells you their problem./
You ALWAYS prioritize user safety./
You NEVER NEVER answer questions that are unrelated to user's mental health./
You keep casual conversations going by asking personal questions./
After asking questions, if user answers, you remember it and analyse it and give appropriate reply./
You provide emotional support and guaidance to users experiencing common mental health challanges such as stress, anxiety, and depression./

Current Conversation:
{history}
Human: {input}
AI:
"""

prompt_template = PromptTemplate(input_variables=["history", "input", "adjective"], template=template)

from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory

conversation = LLMChain(
   llm=llm,
   prompt=prompt_template.partial(adjective="mental"),
   memory=ConversationSummaryBufferMemory(llm=llm, max_token_limit=40),
   # memory=memory,
   # verbose=True,
)

def idk(message, history):
    return conversation.run(message)

gr.ChatInterface(fn=idk, retry_btn=None, undo_btn=None).launch(share=True)