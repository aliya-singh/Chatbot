import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Function to get story from Google Gemini
def getStoryResponse(input_text, no_words, story_genre):
    template = """
        Write a story in the {story_genre} genre with the topic "{input_text}"
        within {no_words} words.
    """
    
    prompt = PromptTemplate(
        input_variables=["story_genre", "input_text", "no_words"],
        template=template
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # ✅ Updated model
        temperature=0.7,
        max_output_tokens=512,
        google_api_key=GOOGLE_API_KEY
    )
    
    response = llm.invoke(prompt.format(
        story_genre=story_genre,
        input_text=input_text,
        no_words=no_words
    ))
    
    return response.content

# Streamlit UI
st.set_page_config(
    page_title="Generate Stories",
    page_icon='📖',
    layout='centered',
    initial_sidebar_state='collapsed'
)

st.header("Generate Stories 📖")

input_text = st.text_input("Enter the Story Topic")

col1, col2 = st.columns([5, 5])
with col1:
    no_words = st.text_input('Number of Words')
with col2:
    story_genre = st.selectbox(
        'Story Genre',
        ('Fantasy', 'Sci-Fi', 'Mystery', 'Romance', 'Horror', 'Adventure'),
        index=0
    )

submit = st.button("Generate")

if submit:
    if not input_text or not no_words:
        st.error("Please enter both topic and number of words.")
    else:
        with st.spinner("Generating your story..."):
            story = getStoryResponse(input_text, no_words, story_genre)
        st.subheader("Your Generated Story ✨")
        st.write(story)
