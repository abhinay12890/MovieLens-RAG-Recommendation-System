from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st


GOOGLE_API_KEY=st.secrets["google_api_key"]

embeddings=HuggingFaceEmbeddings(model='sentence-transformers/all-mpnet-base-v2')

faiss_index=FAISS.load_local(folder_path='movielens_final_embeddings',embeddings=embeddings,allow_dangerous_deserialization=True)

retriver=faiss_index.as_retriever(search_kwargs={"k":50})

st.title("Movie Recommendation Engine")
st.caption("A movie recommendation system based on MovieLens dataset powered by Gemini")

input_text=st.text_input(label="Describe Plot or Keywords")

results=retriver.invoke(input_text)

unique_movies={}

for x in results:
    mid=x.metadata["movie_id"]
    if mid not in unique_movies:
        unique_movies[mid]=x.metadata
unique_movies=list(unique_movies.values())

context=""
for x in unique_movies:
    for key,value in x.items():
        if key in ["Title","Genres"]:
            context+=f"{key}:{value};"
        elif key=="Rating":
            context+=f"{key}:{value}\n"



llm=ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY,model="gemini-2.5-flash")

prompt=ChatPromptTemplate([
    ("system","You are a helpful assistant\n"
     "You will get a string of movies with movie titles, its genres and ratings\n"
     "Your task is to provide top rated movies along with their movie title, genre and rating in descending order of their rating\n"
     "Also generate summary of their movie plot of the received titles\n"
     "Dont Hallicunate only provide Top- 7 rated movie titles based on received user string and generate summary plot based on the top movie titles\n"
     "<string>"
     "{string}"
     "<string"),
     ("user","{context}")
])

output=StrOutputParser()

chain=prompt|llm|output

response=chain.invoke({'string':context,"context":input_text})


if input_text:
    st.write(response)
