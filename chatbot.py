import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.8,
    max_output_tokens=2000,  # Can be increased, but keep API limits in mind
    google_api_key=GOOGLE_API_KEY
)

# Function to generate story
def getStoryResponse(input_text, no_words, story_genre):
    template = """
    You are a master storyteller.

    Write a story in the **{story_genre}** genre on the topic: "{input_text}".
    - Include a creative title in bold on the first line.
    - Word count: Around {no_words} words (strict).
    - Divide story into:
        **Beginning** — hook and setup
        **Middle** — conflict and tension
        **Ending** — satisfying closure
    - Use vivid imagery and emotions.
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

    # Post-process to enforce word count
    story = response.content.strip()
    words = story.split()
    if len(words) > int(no_words):
        story = " ".join(words[:int(no_words)]) + "..."

    return story

# UI
st.header("Generate Stories 📖")

input_text = st.text_input("Enter the Story Topic")
col1, col2 = st.columns([5, 5])
with col1:
    no_words = st.number_input('Number of Words', min_value=50, max_value=2000, step=50)
with col2:
    story_genre = st.selectbox(
        'Story Genre',
        ('Fantasy', 'Sci-Fi', 'Mystery', 'Romance', 'Horror', 'Adventure'),
        index=0
    )

if st.button("Generate"):
    if not input_text or not no_words:
        st.error("Please enter both topic and number of words.")
    else:
        with st.spinner("Generating your story..."):
            story = getStoryResponse(input_text, no_words, story_genre)

        # Extract title if present
        story_lines = story.split("\n")
        if story_lines[0].startswith("**") and story_lines[0].endswith("**"):
            st.markdown(f"## {story_lines[0].strip('**')}")
            story_body = "\n".join(story_lines[1:])
        else:
            story_body = story

        # Show sections in collapsible expanders
        parts = story_body.split("**")
        for part in parts:
            if part.strip().lower().startswith("beginning"):
                with st.expander("🪄 Beginning"):
                    st.markdown(part.replace("Beginning", "").strip())
            elif part.strip().lower().startswith("middle"):
                with st.expander("🔥 Middle"):
                    st.markdown(part.replace("Middle", "").strip())
            elif part.strip().lower().startswith("ending"):
                with st.expander("🎯 Ending"):
                    st.markdown(part.replace("Ending", "").strip())
            else:
                if part.strip():
                    st.markdown(part.strip())

        # Word count badge
        word_count = len(story.split())
        st.info(f"**Word Count:** {word_count} / {no_words}")

        # Download button
        st.download_button(
            label="📥 Download Story as TXT",
            data=story,
            file_name="story.txt",
            mime="text/plain"
        )
