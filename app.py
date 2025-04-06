from flask import Flask, render_template, request, jsonify
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import os
from dotenv import load_dotenv
import pickle

# Load environment variables
load_dotenv()

# Explicitly specify templates and static folders
app = Flask(__name__, template_folder='templates', static_folder='static')

# Initialize conversation chain
def initialize_chain():
    groq_api_key = os.environ.get("GROQ_API_KEY")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.load_local(
        "embeddings",
        embeddings,
        allow_dangerous_deserialization=True
    )

    with open("embeddings/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    print(f"Loaded {metadata['num_documents']} documents with {metadata['num_chunks']} chunks")

    llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama3-70b-8192",
        temperature=0.3,
        max_tokens=4000
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    system_prompt = SystemMessagePromptTemplate.from_template(
        "You are TalentForge AI, a placement and interview expert. First try to answer based on provided context chunks. If no direct context is available or if the context doesn't fully address the question, then use your own knowledge to give the most relevant answer regarding placements, HR queries, and interview guidance. Format your responses using Markdown syntax for better readability. Use tables, code blocks, lists, quotes, and other Markdown features when appropriate."
    )
    human_prompt = HumanMessagePromptTemplate.from_template(
        "Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}\n\nIf the context doesn't fully address the question or if you need to provide additional information, use your own knowledge to give a comprehensive answer."
    )

    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

    return chain

conversation_chain = initialize_chain()
conversation_history = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_message = request.json.get('message', '')

    if not user_message:
        return jsonify({'answer': 'Please ask a question'})

    result = conversation_chain({"question": user_message, "chat_history": conversation_history})
    answer = result["answer"]

    conversation_history.append((user_message, answer))

    return jsonify({'answer': answer})

@app.route('/reset', methods=['POST'])
def reset():
    global conversation_history, conversation_chain
    conversation_history = []

    # Reinitialize the memory for LangChain
    conversation_chain.memory.clear()

    return jsonify({'status': 'success', 'message': 'Conversation reset successfully'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render sets this
    app.run(host='0.0.0.0', port=port)

