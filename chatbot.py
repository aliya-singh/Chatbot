# app.py
import os
import math
import time
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Load env ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Page config ---
st.set_page_config(page_title="AI Story Crafter — Chapter Mode", layout="wide")

# --- Basic checks ---
if not GOOGLE_API_KEY:
    st.error("Google API key not found. Please set GOOGLE_API_KEY in your .env.")
    st.stop()

# --- Initialize LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.8,
    max_output_tokens=1200,
    google_api_key=GOOGLE_API_KEY
)

# --- Session state defaults ---
if "title" not in st.session_state:
    st.session_state.title = ""
if "synopsis" not in st.session_state:
    st.session_state.synopsis = ""
if "chapters" not in st.session_state:
    st.session_state.chapters = []
if "current_chapter" not in st.session_state:
    st.session_state.current_chapter = 0
if "target_total_words" not in st.session_state:
    st.session_state.target_total_words = 0
if "chapter_word_target" not in st.session_state:
    st.session_state.chapter_word_target = 0
if "generating" not in st.session_state:
    st.session_state.generating = False

# --- Helper functions ---
def count_words(text: str) -> int:
    return len(text.split())

def enforce_word_limit(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]) + "..."

def generate_title_and_synopsis(topic: str, genre: str, word_target: int):
    template = """
You are a master storyteller and editor.

TASK:
1) Produce a creative title (one short line).
2) Then write a concise synopsis (1 paragraph, ~100-150 words) for a {genre} story based on the prompt: "{topic}".
3) Keep it intriguing and suitable for a multi-chapter novel of approximately {word_target} words.

Format:
**Title:** <title line>
**Synopsis:** <one paragraph>

Produce only Title and Synopsis in the format above.
"""
    prompt = PromptTemplate(
        input_variables=["genre", "topic", "word_target"],
        template=template
    )
    resp = llm.invoke(prompt.format(genre=genre, topic=topic, word_target=word_target))
    return resp.content.strip()

def generate_first_chapter(title: str, synopsis: str, topic: str, genre: str, chapter_num: int, chapter_word_target: int):
    template = """
You are an experienced novelist.

TASK:
Write Chapter {chapter_num} of a novel titled "{title}" (genre: {genre}), based on this synopsis:

{synopsis}

Chapter length: approximately {chapter_word_target} words (strictly do not exceed this number).
Structure: Start the chapter with a short chapter header line like "Chapter {chapter_num} — <subtitle or short phrase>".
Write in immersive, evocative prose. Keep the tone consistent with the genre.
Make the chapter a complete, satisfying part of the story while leaving room to continue in subsequent chapters.

Begin Chapter {chapter_num} now:
"""
    prompt = PromptTemplate(
        input_variables=["chapter_num", "title", "genre", "synopsis", "chapter_word_target"],
        template=template
    )
    resp = llm.invoke(prompt.format(
        chapter_num=chapter_num,
        title=title,
        genre=genre,
        synopsis=synopsis,
        chapter_word_target=chapter_word_target
    ))
    text = resp.content.strip()
    return enforce_word_limit(text, chapter_word_target)

def generate_next_chapter(story_so_far: str, title: str, synopsis: str, genre: str, chapter_num: int, chapter_word_target: int):
    story_words = story_so_far.split()
    max_context_words = 1200
    if len(story_words) > max_context_words:
        story_context = " ".join(story_words[-max_context_words:])
        story_context = "(Note: the excerpt below is the most recent part of the story.)\n" + story_context
    else:
        story_context = story_so_far

    template = """
You are a skilled novelist continuing a multi-chapter work.

STORY SO FAR (most recent excerpt or full story):
{story_context}

TASK:
Write Chapter {chapter_num} of the novel titled "{title}" (genre: {genre}), continuing naturally from the excerpt above.
Chapter length: approximately {chapter_word_target} words (strictly do not exceed this number).
Start with a header line like "Chapter {chapter_num} — <subtitle or phrase>".
Keep scene continuity, character voice, and pacing consistent.
Make this chapter satisfyingly complete while advancing the narrative.

Begin Chapter {chapter_num} now:
"""
    prompt = PromptTemplate(
        input_variables=["story_context", "chapter_num", "title", "genre", "chapter_word_target"],
        template=template
    )
    resp = llm.invoke(prompt.format(
        story_context=story_context,
        chapter_num=chapter_num,
        title=title,
        genre=genre,
        chapter_word_target=chapter_word_target
    ))
    text = resp.content.strip()
    return enforce_word_limit(text, chapter_word_target)

# --- App UI ---
st.title("✨ AI Story Crafter — Chapter Mode")
st.write("Generate long stories or novels chunk-by-chunk to avoid truncation and get consistent results.")

with st.sidebar:
    st.header("Create a new story")

    topic = st.text_area("Story topic / prompt")
    genre = st.selectbox(
        "Genre",
        ("Select a genre", "Romance", "Drama", "Fantasy", "Sci-Fi", "Mystery", "Horror", "Adventure")
    )
    total_words = st.number_input("Target total words", min_value=500, max_value=200000, step=500)
    chapter_size = st.number_input("Words per chapter", min_value=200, max_value=2000, step=50)
    auto_generate = st.checkbox("Auto-generate all chapters until target reached")

    start_button = st.button("🪄 Start New Story")

if start_button:
    # --- Validation ---
    if not topic.strip():
        st.error("Please provide a story topic/prompt.")
        st.stop()
    if genre == "Select a genre":
        st.error("Please select a genre.")
        st.stop()
    if total_words <= 0:
        st.error("Please enter a valid target total word count.")
        st.stop()
    if chapter_size <= 0:
        st.error("Please enter a valid words per chapter count.")
        st.stop()

    # Reset session state
    st.session_state.title = ""
    st.session_state.synopsis = ""
    st.session_state.chapters = []
    st.session_state.current_chapter = 0
    st.session_state.target_total_words = int(total_words)
    st.session_state.chapter_word_target = int(chapter_size)

    with st.spinner("Generating title & synopsis..."):
        ts_text = generate_title_and_synopsis(topic, genre, total_words)
        title_line = ""
        synopsis_line = ""
        for line in ts_text.splitlines():
            if line.strip().lower().startswith("**title"):
                if ":" in line:
                    title_line = line.split(":", 1)[1].strip()
            elif line.strip().lower().startswith("**synopsis"):
                if ":" in line:
                    synopsis_line = line.split(":", 1)[1].strip()
        if not title_line:
            for ln in ts_text.splitlines():
                if ln.strip():
                    title_line = ln.strip()
                    break
        if not synopsis_line:
            synopsis_line = ts_text.strip().split("\n")[-1].strip()

        st.session_state.title = title_line
        st.session_state.synopsis = synopsis_line

    # Generate first chapter
    st.session_state.current_chapter = 1
    with st.spinner(f"Generating Chapter {st.session_state.current_chapter}..."):
        chap_text = generate_first_chapter(
            st.session_state.title,
            st.session_state.synopsis,
            topic,
            genre,
            st.session_state.current_chapter,
            st.session_state.chapter_word_target
        )
        st.session_state.chapters.append({
            "chapter": st.session_state.current_chapter,
            "text": chap_text,
            "words": count_words(chap_text)
        })

    # Auto-generate rest if selected
    if auto_generate:
        chapters_needed = math.ceil(st.session_state.target_total_words / st.session_state.chapter_word_target)
        progress = st.progress(0)
        for cnum in range(2, chapters_needed + 1):
            st.session_state.current_chapter = cnum
            story_so_far = "\n\n".join([c["text"] for c in st.session_state.chapters])
            with st.spinner(f"Generating Chapter {cnum}..."):
                chap_text = generate_next_chapter(
                    story_so_far,
                    st.session_state.title,
                    st.session_state.synopsis,
                    genre,
                    cnum,
                    st.session_state.chapter_word_target
                )
                st.session_state.chapters.append({
                    "chapter": cnum,
                    "text": chap_text,
                    "words": count_words(chap_text)
                })
            progress.progress(int(((cnum - 1) / chapters_needed) * 100))
            time.sleep(0.8)
        progress.progress(100)
        st.success("Auto-generation complete.")

# --- Main reading / control area ---
col1, col2 = st.columns([1.4, 2.6])

with col1:
    st.subheader("Story Controls")
    st.write(f"**Title:** {st.session_state.title or '—'}")
    st.write(f"**Synopsis:** {st.session_state.synopsis or '—'}")
    st.write(f"**Chapters generated:** {len(st.session_state.chapters)}")
    st.write(f"**Target total words:** {st.session_state.target_total_words or '—'}")
    st.write(f"**Chapter target (words):** {st.session_state.chapter_word_target}")

    if st.button("➕ Generate Next Chapter"):
        if not st.session_state.title:
            st.error("No story in progress. Use 'Start New Story' first.")
        else:
            next_ch_num = len(st.session_state.chapters) + 1
            story_so_far = "\n\n".join([c["text"] for c in st.session_state.chapters])
            st.session_state.generating = True
            with st.spinner(f"Generating Chapter {next_ch_num}..."):
                chap_text = generate_next_chapter(
                    story_so_far,
                    st.session_state.title,
                    st.session_state.synopsis,
                    genre,
                    next_ch_num,
                    st.session_state.chapter_word_target
                )
                st.session_state.chapters.append({
                    "chapter": next_ch_num,
                    "text": chap_text,
                    "words": count_words(chap_text)
                })
            st.session_state.generating = False
            st.success(f"Chapter {next_ch_num} generated.")

    if st.button("🔁 Regenerate Last Chapter"):
        if not st.session_state.chapters:
            st.info("No chapters to regenerate.")
        else:
            st.session_state.chapters.pop()
            next_ch_num = len(st.session_state.chapters) + 1
            story_so_far = "\n\n".join([c["text"] for c in st.session_state.chapters])
            with st.spinner(f"Regenerating Chapter {next_ch_num}..."):
                chap_text = generate_next_chapter(
                    story_so_far,
                    st.session_state.title,
                    st.session_state.synopsis,
                    genre,
                    next_ch_num,
                    st.session_state.chapter_word_target
                )
                st.session_state.chapters.append({
                    "chapter": next_ch_num,
                    "text": chap_text,
                    "words": count_words(chap_text)
                })
            st.success(f"Chapter {next_ch_num} regenerated.")

    def combined_story_text():
        parts = []
        if st.session_state.title:
            parts.append(st.session_state.title)
            parts.append("")
        if st.session_state.synopsis:
            parts.append("Synopsis:")
            parts.append(st.session_state.synopsis)
            parts.append("")
        for c in st.session_state.chapters:
            parts.append(c["text"])
            parts.append("")
        return "\n".join(parts)

    if st.button("📥 Download full story (TXT)"):
        full_text = combined_story_text()
        st.download_button(
            "Click to download",
            data=full_text,
            file_name=(st.session_state.title or "story") + ".txt",
            mime="text/plain"
        )

    total_generated_words = sum([c["words"] for c in st.session_state.chapters])
    st.write(f"**Total generated words:** {total_generated_words}")
    st.progress(min(1.0, total_generated_words / max(1, st.session_state.target_total_words)))

with col2:
    st.subheader("📖 Story Preview")

    if not st.session_state.chapters:
        st.info("No chapters yet. Start a new story or generate Chapter 1.")
    else:
        for ch in st.session_state.chapters:
            header = f"Chapter {ch['chapter']} — {ch['words']} words"
            with st.expander(header, expanded=(ch['chapter'] == len(st.session_state.chapters))):
                st.markdown(ch["text"].replace("\n", "  \n"))

    if st.checkbox("Show combined book text"):
        st.text_area("Full Book", value=combined_story_text(), height=400)

st.markdown("---")
st.write(
    "Tips: Use smaller chapter sizes (800–1200) for consistent pacing. "
    "If you want a very long book (50k+ words), generate chapters gradually or use Auto-generate with patience."
)
