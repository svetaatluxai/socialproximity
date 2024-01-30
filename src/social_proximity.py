from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import tiktoken
from openai import OpenAI
import json
import re


class SocialProximity:
    def __init__(self) -> None:
        self.knowledge = "./data"# 1) specify the source of data 
        self.vectorised_knowledge = "knowledge_vs"#vector store of factual knowledge
        self.vectorised_memory = "memory_vs"#vector store of summaries of past conversations
        openaikey=open("key.txt", 'r').read().strip()
        self.embeddings_model = OpenAIEmbeddings(openai_api_key=openaikey)
        self.client = OpenAI(api_key = open("key.txt", 'r').read().strip())#better export variable
        self.loader=DirectoryLoader
        self.messages =[]
        self.messagesNoRAG=[]
        self.personas_config="config/personas"
        self.script_file="config/scripts"
        self.maxtokens=4000
        self.cp=re.compile("<CONTEXT>[\s\S]*<END CONTEXT>")

    # 2) load data for retriaval: use directory loader here
    def loadData(self, datadir):
        self.loader = DirectoryLoader(datadir)
        docs = self.loader.load()
        print(len(docs))
        return docs
    
    # 3) chunking    
    # here we split by the number of tokens, but we can also split by markdown or other values using other splitters
    # tiktoken is helpful if we need to make sure
    # that the chunk size does not exceed the allowed number of tokens for OpenAI
    def split_documents(self, loadeddata):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=8)
        transformed=text_splitter.split_documents(loadeddata)
        return transformed
    
    # 4) embed: translate all transformed data to embeddings
    # 5) store in a vector store
    def make_knowledge_embeddings(self, chunked_docs, vectorstore_name: str):
        db = FAISS.from_documents(chunked_docs, self.embeddings_model)
        db.save_local(vectorstore_name)#this is to store index locally
        print("New vectorstore created: ", vectorstore_name)
        return db
    
    # checks if local vector store with the given name already exists
    def check_local_knowledge(self, vectorstore_name: str):
        try:
            # this is to open the idex again
            db = FAISS.load_local(vectorstore_name, self.embeddings_model)
            print("Opening existing vectorstore:", vectorstore_name)
            return db
        except:
            FileNotFoundError
            print("local kowledge does not exist")
            return False

# if local vector store already exists, retrieves context from local
        # otherwise creates a new vector store and searches for context
    def retrieve_context(self, query, vectorstore_obj) -> str:
        context=""
        docs = vectorstore_obj.similarity_search(query)
        context=docs[0].page_content
        print("retrieved context ", context)
        return context
    
    def create_new_vectorstore(self, vectorstore_name):
        context=""
        data=self.loadData(self.knowledge)
        chunked = self.split_documents(data)
        db = self.make_knowledge_embeddings(chunked, vectorstore_name)
        return db
    
    # initialise the system message for each other speaker
    def initialize_messages(self, identityprompt, userprompt, messages):
        messages=[]
        messages.append({"role": "system", "content": identityprompt + "You are talking to " + userprompt + " Respond to "+ userprompt})
        print(messages)
        return messages
    

    # for RAG response, add context to system message
    def update_system_message(self,context):
        sysmess=self.messages[0]["content"]
        if not "CONTEXT" in sysmess:
            sysmess=sysmess+"\nUse the following context to reply to user's message: \n"
            sysmess=sysmess+"<CONTEXT>"+context+"<END CONTEXT>"+ " Do not use lists in responses."
        else:
            sysmess=self.cp.sub("<CONTEXT>"+context+"<END CONTEXT>",sysmess)
        print("SYSMESSAGE:\n ",sysmess)
        return sysmess
    
    def count_tokens(self, messages, model="gpt-3.5-turbo-0613"):
        """This is from openai cookbook"""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        tokens_per_message = 3
        tokens_per_name = 1
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
    
    #generate response with RAG and update dialogue history
    def get_response(self, prompt, context):
        sysmessage=self.update_system_message(context)
        self.messages[0]["content"]=sysmessage
        self.messages.append({"role": "user", "content": prompt})
        print("Tokens: ", self.count_tokens(self.messages))
        # check max_tokens, remove the oldest turn if too long
        while self.count_tokens(self.messages)>self.maxtokens:
            print("Message too long")
            self.messages.pop(1)
        #print(self.messages)
        response= self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=250,
            messages=self.messages
        )
        chatmessage = response.choices[0].message
        messagetext=chatmessage.content
        self.messages.append({"role":"assistant","content":messagetext})
        #print(messagetext)
        return messagetext
    
    #generate response without RAG and update dialogue history
    def get_responseNoRAG(self, prompt):
        self.messagesNoRAG.append({"role": "user", "content": prompt})
        print("Tokens no RAG: ", self.count_tokens(self.messagesNoRAG))
        while self.count_tokens(self.messagesNoRAG)>self.maxtokens:
            print("Message too long")
            self.messagesNoRAG.pop(1)
        #print(self.messages)
        response= self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            max_tokens=250,
            messages=self.messagesNoRAG
        )
        chatmessage = response.choices[0].message
        messagetext=chatmessage.content
        self.messagesNoRAG.append({"role":"assistant","content":messagetext})
        #print(messagetext)
        return messagetext
    
    def load_personas(self, configfile):
        with open(configfile, "r", encoding='utf8') as f:
            personas = json.load(f)
        return personas

    def load_script_for_persona(self, who, script_file):
        script={}
        with open(script_file, "r", encoding='utf8') as f:
            allscripts = json.load(f)
        for a in allscripts:
            if "type" in a.keys():
                if a["type"]==who:
                    script=a
        print(a, " ------ from script")
        return script
    
    
    
if __name__ == "__main__":
    identityprompt=open("config/identityprompt.txt", 'r').read()
    sp=SocialProximity()
    # A new vector store named "knowledge_faiss" will be created
    # First we chack if it already exists
    knowledge=sp.check_local_knowledge("knowledge_faiss")
    if knowledge== False:
        knowledge=sp.create_new_vectorstore("knowledge_faiss")
    personas=sp.load_personas(sp.personas_config)
    for p in personas:
        print(p["who"])
        who=p["who"]
        print(p["prompt"])
        userprompt=p["prompt"]
        script=sp.load_script_for_persona(who, sp.script_file)
        print(script)
        sp.messages=sp.initialize_messages(identityprompt, userprompt,sp.messages)
        sp.messagesNoRAG=sp.initialize_messages(identityprompt, userprompt,sp.messagesNoRAG)
        writefile="dialogues/talk-"+who+"-01.json"
        writefilenorag="dialogues/talkNoRAG-"+who+"-01.json"
        keys=script.keys()
        print(keys)
        print("NEW DIALOGUE")
        for k in keys:
            print("KEY: ",k)
            if k!="type":
                response=sp.get_response(script[k], sp.retrieve_context(script[k], knowledge))
                #print(response)
                responseNR=sp.get_responseNoRAG(script[k])
        print("Done!")
        with open(writefile, "w", encoding='utf8') as wf:
            output=json.dumps(sp.messages, indent=4, ensure_ascii=False,)
            wf.write(output)
        with open(writefilenorag, "w", encoding='utf8') as wfn:
            output=json.dumps(sp.messagesNoRAG, indent=4, ensure_ascii=False,)
            wfn.write(output)
    print("completed")
