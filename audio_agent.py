import os
import json
from datetime import datetime

from langchain.chat_models import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.tools import Tool
from langchain.vectorstores.chroma import Chroma
from functools import partial


from langchain.chains.combine_documents import collapse_docs, split_list_of_docs
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.prompts import format_document
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

# from langchain.document_loaders.json_loader import JSONLoader

from openai import AzureOpenAI

from dotenv import load_dotenv
load_dotenv()



azure_api_key_whisper = os.getenv('AZURE_OPENAI_API_KEY_WHISPER')
azure_endpoint_whisper = os.getenv('AZURE_OPENAI_ENDPOINT_WHISPER')
file_path = 'audio_transscript.json'

azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')

client = AzureOpenAI(
    api_key=azure_api_key_whisper,
    azure_endpoint=azure_endpoint_whisper,
    azure_deployment="whisper",
    api_version="2023-09-01-preview",
)

llm = AzureChatOpenAI(
    api_key=azure_api_key,
    api_version="2023-05-15",
    azure_deployment="gpt-35-turbo-16k",
    azure_endpoint=azure_endpoint,
)

def get_transcript(audio_file):
    if not os.path.exists(audio_file):
        audio_file = "./data/" + audio_file
    client.audio.with_raw_response
    return client.audio.transcriptions.create(
        file=open(audio_file, "rb"),            
        model="whisper",
        language="de",
    ).text


def init_transscript_json():

    directory_path = '.\PubAudio'
    audio_dict = {}
    # Iterate over files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Check if the current item is a file (not a subdirectory)
        if os.path.isfile(file_path):
            print(file_path)
            filename = os.path.basename(file_path)
            print(filename)
            audio_dict[filename] = {'transcript': get_transcript(file_path), 'file_path': file_path, 'description': 'xxxxxx'}



    # Open the file in write mode and use json.dump() to write the dictionary to the file
    with open(file_path, 'w') as file:
        json.dump(audio_dict, file)

def load_data_from_json():
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data



# documents

from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain


documents = []


speechs = load_data_from_json()




for key in speechs:
    documents.append(
        Document(
            page_content=speechs[key]['transcript'],
            metadata={
                "source": key,
                "description": speechs[key]['description'],
                "date": datetime.strptime(speechs[key]['date'], "%Y-%m-%d"),
                "author": speechs[key]['author']
            }
        )
    )

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

chain = prompt | llm

document_prompt = ChatPromptTemplate.from_template("""Content: {page_content}                             
Source: {source}
Author: {author}
Date: {date}
Description: {description}""")

def get_chain_input(user_input: str):
    return {"input": user_input, "context": documents}

document_chain = get_chain_input | create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
    document_prompt=document_prompt,
)


audio_speech_tool = Tool(
    name="Speech Summary",
    func=document_chain.invoke,
    description="Use this tool to do a summary of speeches.",
)