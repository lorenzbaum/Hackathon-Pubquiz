# Hackathon-Pubquiz

In this repository you can find starter code and data for the Pubquiz Hackathon!

Please create a virtual environment and install the requirements.txt:

```
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
```

Make sure to create a .env file with 

```
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY_WHISPER=
AZURE_OPENAI_ENDPOINT_WHISPER=
```

which will be provided for you.

# Data

PubAudio are audio files which might be relevnat for the Quiz

PubDatabase is a working Chroma database to be loaded. Note: You need to use the embeddings provided in quiz.py, or the database will not load.

PubTexts are texts which might be relevnat for the Quiz
