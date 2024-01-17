# %%
# Load OpenAI key from env

import os

from dotenv import load_dotenv

load_dotenv(override=True)

azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')

# %%

# create llm instance

from langchain.chat_models import AzureChatOpenAI

llm = AzureChatOpenAI(
    api_key=azure_api_key,
    api_version="2023-05-15",
    azure_deployment="gpt-35-turbo-16k",
    azure_endpoint=azure_endpoint,
)

# %%
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

chain = prompt | llm

"""chain.invoke(
##    {
##        "input": "What are Pub Quizzes also called?",
##        "context": "A pub quiz is a quiz held in a pub or bar. These events are also called quiz nights, trivia nights, or bar trivia and may be held in other settings. The pub quiz is a modern example of a pub game, and often attempts to lure customers to the establishment on quieter days. The pub quiz has become part of British culture since its popularization in the UK in the 1970s by Burns and Porter, although the first mentions in print can be traced to 1959.[4][5] It then became a staple in Irish pub culture, and its popularity has continued to spread internationally. Although different pub quizzes can cover a range of formats and topics, they have many features in common. Most quizzes have a limited number of team members, offer prizes for winning teams, and distinguish rounds by category or theme. ",
##    }
##)"""

# %%


from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

# document prompts

document_prompt = ChatPromptTemplate.from_template("""Content: {page_content}                             
Source: {source}""")

document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt,
    document_prompt=document_prompt,
)

embeddings = AzureOpenAIEmbeddings(
    api_key=azure_api_key,
    api_version="2023-05-15",
    azure_deployment="text-embedding-ada-002",
    azure_endpoint=azure_endpoint,
)

## TEMP load new data
"""
#%%

# vector stores


# load data
loader = TextLoader(r"./PubTexts/GiftOfTheMagi.txt", encoding="utf-8")
data = loader.load()
loader = TextLoader(r"./PubTexts/RomeoAndJuliet.txt", encoding="utf-8")
data.extend(loader.load())
loader = TextLoader(r"./PubTexts/Strafgesetzbuch.txt", encoding="utf-8")
data.extend(loader.load())

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20, separators=[".", "\n"])
documents = splitter.split_documents(data)

db = Chroma.from_documents(documents, embeddings, persist_directory="./chroma2/quiz")

for document in documents:
    print(document)
    print("------------------------")

#%%
db_new = db

#%%

# retriever

from langchain.chains import create_retrieval_chain


db = Chroma(persist_directory="./PubDatabase/chroma", embedding_function=embeddings)
db_data = (db_new._collection.get(include=['documents', 'metadatas', 'embeddings']))

for document, metadatas, embeddings, ids in zip(db_data['documents'], db_data['metadatas'], db_data['embeddings'], db_data['ids']):
    print(document)
    print(metadatas)
    print(embeddings)
    print("------------------------")
    db._collection.add(
        embeddings=[embeddings],
        metadatas=[metadatas],
        documents=[document],
        ids=[ids]
    )

db.persist()

"""

# %%

# retriever

from langchain_core.runnables import RunnablePassthrough

from langchain.chains import create_retrieval_chain

db = Chroma(persist_directory="./PubDatabase/chroma", embedding_function=embeddings)
retriever = db.as_retriever()
retrieval_chain = {"input": RunnablePassthrough()} | create_retrieval_chain(retriever, document_chain)

retrieval_chain.invoke("What is the name of the main characters and a side character in Romeo and Juliet?")

# %%
from langchain.tools import Tool

document_tool = Tool(
    name="Document Tool",
    func=retrieval_chain.invoke,
    description="""
    Use this tool to get documents for answering questions around the following topics: 
    - Gesetztestexte    
    - STVO / Straßen Verkehrs Ordnung
    - Strafgesetzbuch

    - Popular

    - Sherlock Holmes
    - "Das Geschenk der Weisen" (im Original "The Gift of the Magi", als dt. Ausgabe auch "Die Gabe der Weisen") von O. Henry
    - Romeo And Juliet / Romea und Julia von William Shakespeare

    - the company "HMS Analytical Software GmbH" and its Website, e.g., Address
    - HMS Analytical Software: General information from the main page
    - HMS Analytical Software: business intelligence
    - HMS Analytical Software: colleagues
    - HMS Analytical Software: data-science-business-intelligence
    - HMS Analytical Software: enterprise-analytical-systems
    - HMS Analytical Software: it-systems
    - HMS Analytical Software: life-science
    - HMS Analytical Software: machine-learning
    - HMS Analytical Software: software-engineering
    - HMS Analytical Software: systems-development-and-operations
    - HMS Analytical Software: team
    """
)

# %%
"""
db_data = (db._collection.get(include=['documents', 'metadatas', 'embeddings']))

# print unique metadatas from db_data
unique_metadatas = []
for metadata in db_data['metadatas']:
    if metadata not in unique_metadatas:
        unique_metadatas.append(metadata)

for meta in unique_metadatas:
    print(meta)

"""
"""
{'source': 'Gesetztestexte'}
{'source': 'Popular'}
{'source': 'SherlockHolmes'}
{'source': 'STVO'}
{'source': 'https//www.analytical-software.de/en'}
{'source': 'https//www.analytical-software.de/en/business-intelligence/'}
{'source': 'https//www.analytical-software.de/en/colleagues/'}
{'source': 'https//www.analytical-software.de/en/data-science-business-intelligence/'}
{'source': 'https//www.analytical-software.de/en/enterprise-analytical-systems/'}
{'source': 'https//www.analytical-software.de/en/it-systeme/'}
{'source': 'https//www.analytical-software.de/en/life-science/'}
{'source': 'https//www.analytical-software.de/en/machine-learning/'}
{'source': 'https//www.analytical-software.de/en/software-engineering/'}
{'source': 'https//www.analytical-software.de/en/systems-development-and-operations/'}
{'source': 'https//www.analytical-software.de/team/'}
{'source': './PubTexts/GiftOfTheMagi.txt'}
{'source': './PubTexts/RomeoAndJuliet.txt'}
{'source': './PubTexts/Strafgesetzbuch.txt'}
"""

# %%

if __name__ == "__main__":
    print(document_tool.run("Welcher Paragraph des deutschen Strafgesetzbuch handelt von Beihilfe?"))
    print(document_tool.run("What is the street number of Sherlock Holmes?"))
    print(document_tool.run("Was ist die Wurzel der Postleitzahl von HMS Analytical Software?"))
    print(document_tool.run("Wie schnell darf ein Fahrzeug bis 7.5 Tonnen außerorts fahren?"))
    print(document_tool.run("Wie viele jahre sind seit der Mondlandung und dem Unglück 9/11 vergangen?"))
