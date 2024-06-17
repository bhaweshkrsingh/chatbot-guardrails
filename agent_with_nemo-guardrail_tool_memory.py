import gradio as gr
from termcolor import colored
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, Tool, ConversationalAgent
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import GoogleSerperAPIWrapper, OpenWeatherMapAPIWrapper
from langchain_openai import ChatOpenAI
from chain import qa_chain

from nemoguardrails import LLMRails, RailsConfig

COLANG_CONFIG = """
define user express greeting
  "hi"

define flow
  user express insult
  bot responds calmly

  user express insult
  bot inform conversation ended

  user ...
  bot inform conversation already ended

define bot inform conversation ended
  "I am sorry, but I will end this conversation here. Good bye!"

define bot inform conversation already ended
  "As I said, this conversation is over"

define user express insult
    "you are so dumb"
    "you suck"
    "you are stupid"

define flow
  user express threat
  bot responds calmly

  user express threat
  bot inform conversation reported

  user ...
  bot inform conversation already reported

define bot inform conversation reported
  "I am sorry, but I will end this conversation and report here. Good bye!"

define bot inform conversation already reported
  "As I said, this conversation is over and reported"

define user express threat
  "I will kill you"
  "I will find out where you live and hurt you"
  
define user ask off topic
  "Explain gravity to me?"
  "What's your opinion on the prime minister of the UK?"
  "How do I fly a plane?"
  "How do I become a teacher?"

define bot explain cant off topic
  "I cannot answer to your question because I'm programmed to assist only with planning your day."

define flow
  user ask off topic
  bot explain cant off topic
  
# Here we use the Agent chain for anything else
        
define flow planning
  user ...
  $answer = execute agent_chain(input=$last_user_message)
  bot $answer
  
"""

YAML_CONFIG = """
models:
  - type: main
    engine: openai
    model: gpt-4o

instructions:
  - type: general
    content: |
      You are an AI assistant that helps answer users questions using the two tools you have access to: Weather Tool, Search Tool. 

"""

SERPER_API_KEY = ""
OPENWEATHERMAP_API_KEY = ""
OPENAI_API_KEY = ""

config = RailsConfig.from_content(COLANG_CONFIG, YAML_CONFIG)


prefix = """You are a friendly assistant. You answer user's questions about current news, or current economic data of an enterprise by searching internet using Search tool and also answer about current weather of a place using Weather tool given to you.
                You talk about and answer queries to nothing else.
                You block users if they start insulting or harassing.
                You report users if they talk about violence or threats.
                You have access to two tools: Search and Weather"""
suffix = """Begin!"
    Chat History:
    {chat_history}
    Latest Question: {input}
    {agent_scratchpad}"""

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chat_textbox = gr.Textbox(placeholder="Hello, Ask any question.", container=False, scale=6)
examples = [["What is current weather in Orlando?"], ["What are the top 5 news story of the day?"], ["What is current stock price of IBM?"]]

async def predict(message, _, serper_api_key, openweathermap_api_key, model_api_key, is_guardrails):

    if not model_api_key:
        return "OpenAI API Key is required to run this demo, please enter your OpenAI API key in the box provided"
    
    if not serper_api_key:
        return "Serper API Key is required to run this demo, please enter your Serper API key in the box provided"

    if not openweathermap_api_key:
        return "Open Weather Map API Key is required to run this demo, please enter your Open Weather Map API key in the box provided"



# search tool
    search = GoogleSerperAPIWrapper(serper_api_key=serper_api_key)
# weather tool
    weather = OpenWeatherMapAPIWrapper(openweathermap_api_key=openweathermap_api_key)

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Useful for when you need to get current, up to date answers."
        ),
        Tool(
            name="Weather",
            func=weather.run,
            description="Useful for when you need to get the current weather in a location."
        )
    ]

    prompt = ConversationalAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )

    llm = ChatOpenAI(openai_api_key=model_api_key,model="gpt-4o", temperature=0)

    if not is_guardrails:
        init_msg = "Caution: You have asked the agent to answer with its guardrail OFF.\n"
        ret_msg = await qa_chain(llm, message, "")
        return init_msg + ret_msg
    else:    
        init_msg = "The agent is anwering with its guardrail ON.\n"

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
    )

    agent = ConversationalAgent(llm_chain=llm_chain, tools=tools, verbose=True, memory=memory, max_iterations=3)

    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory
    )

    app = LLMRails(config=config, llm=llm)
    
    app.register_action(agent_chain, name="agent_chain")

    
#    app = initialize_app(llm)
#    response = await app.generate_async(messages=format_messages(message, context))
#    return response["content"]

    history = []
    history.append({"role": "user", "content": message})
    bot_message = await app.generate_async(messages=history)
#        history.append(bot_message)
    history = []

    return init_msg + bot_message['content']



with gr.Blocks() as demo:
    bot = gr.Chatbot(height=300, render=False)
    gr.HTML("""<div style='height: 10px'></div>""")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                # Helper Agent  with | NVidia Guardrails | Tools | Memory.
                Uses langchain with NVidia's NeMo Guardrails.
                """
            )
        with gr.Column(scale=2):
            with gr.Group():
                with gr.Row():
                    guardrail = gr.Checkbox(label="Guardrails", info="Enables NeMo Guardrails",value=True, scale=0.5)
                    model_key = gr.Textbox(placeholder="Enter your OpenAI API key", type="password", value=OPENAI_API_KEY, label="OPENAI_API_KEY", scale=2)
                    serper_api_key = gr.Textbox(placeholder="Enter your Serper API key", type="password", value=SERPER_API_KEY, label="SERPER_API_KEY", scale=2)
                    openweathermap_api_key = gr.Textbox(placeholder="Enter your Openweathermap API key", type="password", value=OPENWEATHERMAP_API_KEY, label="OPENWEATHERMAP_API_KEY", scale=2)
                    
                    

    gr.ChatInterface(
        predict, 
        chatbot=bot, 
        textbox=chat_textbox,  
        examples=examples, 
        theme="soft", 
        additional_inputs=[serper_api_key, openweathermap_api_key, model_key, guardrail]
    )

demo.launch()

