import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure page
st.set_page_config(page_title="AI Story Crafter", layout="wide")

# Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.8,
    max_output_tokens=2000,
    google_api_key=GOOGLE_API_KEY
)

# Session storage for story vault
if "story_vault" not in st.session_state:
    st.session_state.story_vault = []

# Function to generate story
def getStory(input_text, no_words, story_genre):
    template = """
    You are a master storyteller.

    Write a {story_genre} story based on: "{input_text}".
    - Include a creative title in bold on the first line.
    - Word count: Around {no_words} words.
    - Structure: **Beginning**, **Middle**, **Ending**.
    - Use vivid sensory details and emotions.
    - Make it unique and immersive.
    """
    prompt = PromptTemplate(
        input_variables=["story_genre", "input_text", "no_words"],
        template=template
    )

    response = llm.invoke(prompt.format(
        story_genre=story_genre,
        input_text=input_text,
        no_words=no_words
    ))

    story = response.content.strip()

    # Enforce word count
    words = story.split()
    if len(words) > int(no_words):
        story = " ".join(words[:int(no_words)]) + "..."

    return story

# Sidebar: App Settings
st.sidebar.header("🛠 Story Settings")
mode = st.sidebar.radio("Select Mode", ["Short Story", "Novel Mode"])
story_genre = st.sidebar.selectbox(
    "Story Genre",
    ('Fantasy', 'Sci-Fi', 'Mystery', 'Romance', 'Horror', 'Adventure')
)
no_words = st.sidebar.slider("Word Count", 50, 2000, 500, step=50)
input_text = st.sidebar.text_area("Story Topic", placeholder="Enter your story idea here...")
generate_btn = st.sidebar.button("✨ Generate Story")

# Main Layout
col1, col2 = st.columns([1.2, 2])

# Story Generation
if generate_btn:
    if not input_text:
        st.sidebar.error("Please enter a story topic.")
    else:
        with st.spinner("Crafting your story..."):
            story = getStory(input_text, no_words, story_genre)

        # Save in vault
        st.session_state.story_vault.append(story)

        # Display story
        with col2:
            story_lines = story.split("\n")
            if story_lines[0].startswith("**") and story_lines[0].endswith("**"):
                st.markdown(f"<h1 style='text-align:center;'>{story_lines[0].strip('**')}</h1>", unsafe_allow_html=True)
                story_body = "\n".join(story_lines[1:])
            else:
                story_body = story

            parts = story_body.split("**")
            for part in parts:
                if part.strip().lower().startswith("beginning"):
                    with st.expander("🪄 Beginning", expanded=True):
                        st.write(part.replace("Beginning", "").strip())
                elif part.strip().lower().startswith("middle"):
                    with st.expander("🔥 Middle", expanded=True):
                        st.write(part.replace("Middle", "").strip())
                elif part.strip().lower().startswith("ending"):
                    with st.expander("🎯 Ending", expanded=True):
                        st.write(part.replace("Ending", "").strip())
                else:
                    if part.strip():
                        st.write(part.strip())

            st.info(f"**Word Count:** {len(story.split())} / {no_words}")

            # Download options
            st.download_button("📥 Download as TXT", data=story, file_name="story.txt", mime="text/plain")

# Left Column: Story Vault
with col1:
    st.subheader("📚 Story Vault")
    if st.session_state.story_vault:
        for idx, saved_story in enumerate(reversed(st.session_state.story_vault), 1):
            st.markdown(f"**Story {idx}:** {saved_story.split()[0:5]}...")
            if st.button(f"View Story {idx}"):
                st.write(saved_story)
    else:
        st.info("No stories yet. Generate one to save it here!")
