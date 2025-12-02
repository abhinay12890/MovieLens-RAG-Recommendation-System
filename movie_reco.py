from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st


GOOGLE_API_KEY=st.secrets["google_api_key"]

embeddings=HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')

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

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You will receive a list of movies with titles, genres, and ratings.\n"
     "Your task:\n"
     "1. Extract all valid movies.\n"
     "2. Sort them by rating (descending).\n"
     "3. Return ONLY the top 7.\n\n"

     "OUTPUT FORMAT (STRICT):\n"
     "For each movie, you must output exactly three lines:\n"
     "Line 1: {{Movie Title}} ({{Year}}) | Genres: {{Genres}} | Rating: {{Rating}}\n"
     "Line 2: {{2–3 short summary lines}}\n"
     "Line 3: (Empty Line)\n"
     "\n"

     "CRITICAL FORMATTING RULES:\n"
     "- METADATA SEPARATORS: Use a vertical bar ' | ' to separate Title, Genres, and Rating.\n"
     "- NEWLINE ENFORCEMENT: The summary MUST NOT appear on Line 1. It must start on Line 2.\n"
     "- No commas after the movie title.\n"
     "- No hallucinations.\n\n"

     "Movies received:\n{movies}"
    ),
    ("user", "User query: {query}")
])



output=StrOutputParser()

chain=prompt|llm|output

response=chain.invoke({'movies':context,"query":input_text})


if input_text:
    st.write(response)

footer = st.container()
with footer:
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: grey; font-size: 14px;'>"
        "Developed by <b>Kalavakuri Abhinay</b>"
        "</div>",
        unsafe_allow_html=True
    )
