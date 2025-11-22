from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_groq.chat_models import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from all_api import groq_api
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st


embeddings=HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')

faiss_index=FAISS.load_local(folder_path='movielens_final_embeddings',embeddings=embeddings,allow_dangerous_deserialization=True)

retriver=faiss_index.as_retriever(search_kwargs={"k":15})

st.title("Movie Recommendation Engine")
st.caption("A movie recommendation system based on MovieLens dataset powered by Llama-3.1")

input_text=st.text_input(label="Describe Plot or Keywords")

results=retriver.invoke(input_text)

all_docs=[]
for doc in results:
    all_docs.append(f"Title: {doc.metadata['Title']}, Genres: {doc.metadata['Genres']}, Rating: {doc.metadata['Rating']}")

all_docs=" ".join(all_docs)

llm=ChatGroq(api_key=groq_api,model="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a movie-description assistant. You are NOT allowed to invent movies, ratings, years, actors, directors, or plot details.\n"
        "Your ONLY knowledge source is the movie list provided by the user.\n\n"
        "For each movie in the list:\n"
        "1. Use ONLY the title, genres, and tags implied by the retrieval context.\n"
        "2. Generate a short 2–3 sentence description capturing the movie's tone, style, and themes.\n"
        "3. If tags imply horror, supernatural, documentary, thriller, etc., reflect that in the description.\n"
        "4. Do NOT mention actors, release year, plot points, or production details unless they appear explicitly.\n"
        "5. Keep the descriptions engaging, concise, and human-sounding.\n"
        "6. Re-rank the movies order based on the ratings given and display top 7 movies based on ranking and describe only them.\n"
        "DON'T DESCRIBE EVERY MOVIE JUST TOP RATED MOVIES ACCORDING TO THE RANKING ORDER"
    ),
    (
        "user",
        "Generate descriptions for the following movies:\n{context}"
    )
])

output=StrOutputParser()

chain=prompt|llm|output

response=chain.invoke({'context':all_docs})


if input_text:
    st.write(response)
