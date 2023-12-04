import time
import streamlit as st
from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="Assistant",
    page_icon=":robot:"
)

st.write("### Welcome to LLM Bot!")

st.write(
    "You can ask me anything about LLM's! So excited to answer your questions :)")

streaming=True

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

@st.cache_resource()
def load_llm():
    # load the llm with ctransformers
    llm = CTransformers(model='model/llama-2-7b-chat.ggmlv3.q2_K.bin', # model available here: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
                    model_type='llama',
                    config={'max_new_tokens': 256, 'temperature': 0})
    return llm

@st.cache_resource()
def load_vector_store():

    # load the vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'})
    db = FAISS.load_local("faiss", embeddings)
    return db

@st.cache_data()
def load_prompt_template():

    # prepare the template we will use when prompting the AI
    template = """Use the provided context to answer the user's question.
    If you don't know the answer, respond with "I do not know.".

    Context: {context}
    History: {history}
    Question: {question}
    Answer:
    """

    prompt = PromptTemplate(
                            template=template,
                            input_variables=['context', 'question']
                            )

    return prompt 

def create_qa_chain():

    # load the llm, vector store, and the prompt
    llm = load_llm()
    db = load_vector_store()
    prompt = load_prompt_template()

    # create the qa_chain
    retriever = db.as_retriever(search_kwargs={'k': 2})
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=retriever,
                                        return_source_documents=True,
                                        chain_type_kwargs={'prompt': prompt,
                                                          "memory": ConversationBufferMemory(
                                                                        memory_key="history",
                                                                        input_key="question")})
    
    return qa_chain

def generate_response(query, qa_chain):
    # use the qa_chain to answer the given query
    return qa_chain({'query':query})['result']

def get_user_input():
    # get the user query
    input_text = st.text_input('Ask me anything about Large Language Models!', "", key='input')
    return input_text

# create the qa_chain
qa_chain = create_qa_chain()
prompt = st.chat_input("Type a message...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    if streaming:
        res = generate_response(query=prompt, qa_chain=qa_chain)
        with st.chat_message("Assistant"):
            message_placeholder = st.empty()
            full_response = ""

            buffer = ""
            for chunk in res.split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            #message_placeholder.markdown(res + "▌")
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
    else:
        pass
