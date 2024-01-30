# socialproximity

This is the code used for the study of social proximity cues in language generated by ChatGPT.
The study was accepted for publication at TEICAI@EACL2024 workshop and will be presented in Maltta on March 22.
Workshop website: https://sites.google.com/view/teicai2024/

Publication title: Non-Referential Functions of Language in Social Agents: The Case of Social Proximity
Author: Dr. Sviatlana Höhn (LuxAI S. A. Luxembourg, Luxembourg)

## Prerequisites

Install the following libraries first:
- langchain
- langchain_community
- langchain-openai
- openai
- unstructured
- tiktoken
- faiss-cpu 

You will need your OpenAI API access token to run this code. 

## Knowledge used for publication

The knowledge used for RAG is generated by OpenAI and is stored in the `data` folder. When run for the first time, the program will generate embeddings from the knowledge and store them in a vector store with a default name faiss_knowledge. If you change the documents and want to re-generate embeddings, you will need to provide a diffeernt name for the vector store OR delete the local copy of the vector store.
Indexing is not implemented, but can be added if needed.
If you run this code multiple times, it will try to use a local copy of the vector store (if already exists).

## Data analysed in the publication

The folder dialogues contains the generated dialogues between artificial characters used for the analysis in the paper. 
