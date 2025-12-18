from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st


GOOGLE_API_KEY=st.secrets["google_api_key"]

embeddings=HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')

faiss_index=FAISS.load_local(folder_path='faiss_index',embeddings=embeddings,allow_dangerous_deserialization=True)

retriver=faiss_index.as_retriever(search_kwargs={"k":20})

st.title("🎬 CineSense")
st.caption("Discover personalized movie recommendations powered by AI and the MovieLens dataset.")

input_text = st.text_input("Search for a movie vibe:", placeholder="e.g., sad romance movies")

results=retriver.invoke(input_text)

ans=""
for x in results:
    ans+= f"Title: {x.metadata['title']}, Genres: {x.metadata['genres']}, Avg. Rating: {x.metadata['rating']:.2f}"+ "\n"


llm=ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY,model="gemini-2.5-flash")

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict data formatting engine for a Streamlit app. You will receive a list of movies.\n"
     "Your task is to extract valid movies, sort by rating (descending), and return the top 7.\n\n"

     "### OUTPUT RULES (CRITICAL)\n"
     "1. **NO PREAMBLE:** Output ONLY the list.\n"
     "2. **CARD FORMAT:** Use a horizontal rule `---` to separate every movie.\n"
     "3. **NEWLINES:** You MUST use double newlines (`\\n\\n`) between the Title, Metadata, and Summary.\n"
     "4. **SUMMARY LENGTH:** Ignore short taglines. Generate a compelling **1-2 sentence summary** describing the plot for every movie.\n\n"

     "### REQUIRED FORMAT EXAMPLE\n"
     "Follow this exact Markdown pattern:\n\n"
     "**Titanic** (1997) :star: 7.8\n\n"
     "_Genres: Romance, Drama_\n\n"
     "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic. Their forbidden romance unfolds against the backdrop of the ship's maiden voyage.\n\n"
     "---\n\n"
     "**The Matrix** (1999) :star: 8.7\n\n"
     "_Genres: Action, Sci-Fi_\n\n"
     "A computer hacker learns from mysterious rebels about the true nature of his reality. He joins the war against the controllers to free humanity from the simulated world.\n\n"
     "---\n\n"
     "(End of Example)\n\n"

     "Now, process the following movies:\n{movies}"
    ),
    ("user", "{query}")
])


output=StrOutputParser()

chain=prompt|llm|output

if input_text:
    with st.spinner("Finding recommendations..."):
        response = chain.invoke({'movies': ans, "query": input_text})
        st.write(response)
else:
    pass

footer = st.container()
with footer:
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: grey; font-size: 14px;'>"
        "Developed by <b>Kalavakuri Abhinay</b>"
        "</div>",
        unsafe_allow_html=True
    )
