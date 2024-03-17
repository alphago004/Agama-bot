from flask import Flask, render_template, request
import os 
from apikey import apikey 
import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

app = Flask(__name__)

# Setup OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey

# Define Streamlit app framework
@app.route('/')
def index():
    return render_template('/index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Get user input from the form
    prompt = request.form['prompt']

    # Define Prompt templates
    title_template = PromptTemplate(
        input_variables = ['topic'], 
        template='write me a youtube video title about {topic}'
    )

    script_template = PromptTemplate(
        input_variables = ['title', 'wikipedia_research'], 
        template='write me a youtube video script based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research} '
    )

    # Define Memory 
    title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

    # Initialize Llms+9
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
    script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

    wiki = WikipediaAPIWrapper()

    # Generate title and script based on user input
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    # Render result template with generated title and script
    return render_template('result.html', title=title, script=script, wiki_research=wiki_research)

if __name__ == '__main__':
    app.run(debug=True)
