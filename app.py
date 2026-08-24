import json
import html
import os
from io import BytesIO

import requests
import streamlit as st

from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag_chatbot import rag_query


# ============================================================
# 1. ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found in .env"
    )


# ============================================================
# 2. CONFIGURATION
# ============================================================

GENERATION_MODEL = "gemini-3.6-flash"


# ============================================================
# 3. INITIALIZE GEMINI CLIENT
# ============================================================

intent_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# 4. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="READORA",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 5. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .readora-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }

    .readora-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .readora-subtitle {
        font-size: 1.1rem;
        opacity: 0.7;
        margin-bottom: 1rem;
    }

    .book-title {
        font-size: 1.05rem;
        font-weight: 700;
        line-height: 1.35;
        margin-top: 0.6rem;
    }

    .book-author {
        font-size: 0.9rem;
        opacity: 0.7;
        margin-top: 0.2rem;
    }

    .book-rating {
        font-weight: 600;
        margin-top: 0.4rem;
    }

    .book-cover-placeholder {
        height: 300px;
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        font-size: 3rem;
        background: rgba(128, 128, 128, 0.06);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 6. SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================================
# 7. INTENT CLASSIFICATION
# ============================================================

def classify_intent(user_message):
    """
    Gemini determines the user's intent.

    The application does not manually search for words
    such as 'hello', 'hi', 'اهلا', etc.
    """

    prompt = f"""
You are the intent router for READORA, an intelligent
book recommendation assistant.

Classify the user's message into EXACTLY ONE of these intents:

1. greeting
   The user is greeting, welcoming, thanking, or starting
   a conversation without requesting book information.

2. book_recommendation
   The user wants book recommendations, suggestions,
   discovery, or help choosing what to read.

3. book_information
   The user asks about a specific book, author, genre,
   plot, rating, publication year, or other book-related
   information.

4. general_conversation
   The user is talking casually and does not need book retrieval.

5. other
   Anything that does not fit the categories above.

Examples:

hello
hi
hey
اهلا
أهلاً
مرحبا
السلام عليكم
شكرا
thank you

=> greeting

"I want a fantasy book about magic"
=> book_recommendation

"Recommend a mystery novel"
=> book_recommendation

"What is The Changeling Sea about?"
=> book_information

"Who wrote Harry Potter?"
=> book_information

"How are you?"
=> general_conversation

Return ONLY JSON in this exact structure:

{{
    "intent": "greeting"
}}

User message:
{user_message}
""".strip()

    response = intent_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "OBJECT",
                "properties": {
                    "intent": {
                        "type": "STRING",
                        "enum": [
                            "greeting",
                            "book_recommendation",
                            "book_information",
                            "general_conversation",
                            "other"
                        ]
                    }
                },
                "required": [
                    "intent"
                ]
            }
        )
    )

    if not response.text:
        return "other"

    try:

        result = json.loads(
            response.text
        )

        intent = result.get(
            "intent",
            "other"
        )

        valid_intents = {
            "greeting",
            "book_recommendation",
            "book_information",
            "general_conversation",
            "other"
        }

        if intent in valid_intents:
            return intent

    except (
        json.JSONDecodeError,
        AttributeError,
        TypeError
    ):
        pass

    return "other"


# ============================================================
# 8. GENERATE NON-RETRIEVAL RESPONSE
# ============================================================

def generate_non_retrieval_response(
    user_message,
    intent
):
    """
    Generate a natural response when Pinecone retrieval
    is not required.
    """

    if intent == "greeting":

        prompt = f"""
You are READORA, a friendly book recommendation assistant.

The user is simply greeting you.

Reply naturally and briefly.
Do NOT recommend books yet.
Invite the user to tell you what kind of book they are
looking for.

User message:
{user_message}
""".strip()

    elif intent == "general_conversation":

        prompt = f"""
You are READORA, a friendly book recommendation assistant.

Reply naturally to the user's casual conversation.

Do not invent book recommendations unless the user asks
for them.

User message:
{user_message}
""".strip()

    else:

        prompt = f"""
You are READORA, a book recommendation assistant.

The user's message does not contain enough information
to perform book retrieval.

Reply naturally and ask for clarification when useful.

User message:
{user_message}
""".strip()

    response = intent_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt
    )

    if response.text:
        return response.text.strip()

    return (
        "Hello! 👋 I'm READORA. "
        "Tell me what kind of book you're looking for."
    )


# ============================================================
# 9. CLEAN IMAGE URL
# ============================================================

def clean_image_url(url):
    """
    Convert possible Markdown image/link format into
    a direct URL.

    Example:
    [https://example.com/a.jpg](https://example.com/a.jpg)

    becomes:
    https://example.com/a.jpg
    """

    if url is None:
        return ""

    url = str(url).strip()

    if not url:
        return ""

    if url.lower() in {
        "nan",
        "none",
        "null"
    }:
        return ""

    # --------------------------------------------------------
    # Markdown link
    # --------------------------------------------------------

    if url.startswith("["):

        start = url.find("](")
        end = url.rfind(")")

        if (
            start != -1
            and end != -1
        ):

            url = url[
                start + 2:end
            ].strip()

    return url


# ============================================================
# 10. LOAD IMAGE
# ============================================================

def load_book_image(image_url):
    """
    Download the image from the external source and return
    the bytes as BytesIO so Streamlit can display it.

    Returns:
        BytesIO | None
    """

    image_url = clean_image_url(
        image_url
    )

    if not image_url:
        return None

    if not image_url.startswith(
        ("http://", "https://")
    ):
        return None

    try:

        response = requests.get(
            image_url,
            timeout=10,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/151.0 Safari/537.36"
                ),
                "Accept": (
                    "image/avif,image/webp,image/apng,"
                    "image/svg+xml,image/*,*/*;q=0.8"
                ),
            }
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        # Make sure the response is actually an image.
        if not content_type.startswith(
            "image/"
        ):
            return None

        return BytesIO(
            response.content
        )

    except Exception as e:

        print(
            f"Image loading failed: "
            f"{image_url}"
        )

        print(
            f"Reason: {e}"
        )

        return None


# ============================================================
# 11. DISPLAY IMAGE
# ============================================================

def display_book_image(image_url):

    image_data = load_book_image(
        image_url
    )

    if image_data is not None:

        try:

            st.image(
                image_data,
                use_container_width=True
            )

            return

        except Exception as e:

            print(
                f"Streamlit image display failed: "
                f"{e}"
            )


    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="book-cover-placeholder">
            📖
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 12. DISPLAY BOOK CARD
# ============================================================

def display_book_card(book):

    title = str(
        book.get(
            "title",
            "Unknown Title"
        )
    )

    author = str(
        book.get(
            "author_name",
            "Unknown Author"
        )
    )

    image_url = clean_image_url(
        book.get(
            "image_url",
            ""
        )
    )

    rating = book.get(
        "average_rating",
        ""
    )

    genres = str(
        book.get(
            "genres",
            ""
        ) or ""
    )


    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    display_book_image(
        image_url
    )


    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="book-title">
            {html.escape(title)}
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Author
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="book-author">
            by {html.escape(author)}
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # Rating
    # --------------------------------------------------------

    if (
        rating is not None
        and str(rating).strip()
        not in {
            "",
            "nan",
            "None"
        }
    ):

        try:

            rating_value = float(
                rating
            )

            st.markdown(
                f"""
                <div class="book-rating">
                    ⭐ {rating_value:.2f}
                </div>
                """,
                unsafe_allow_html=True
            )

        except (
            ValueError,
            TypeError
        ):
            pass


    # --------------------------------------------------------
    # Genres
    # --------------------------------------------------------

    if genres:

        st.caption(
            genres
        )


# ============================================================
# 13. HEADER
# ============================================================

st.markdown(
    """
    <div class="readora-header">

        <div class="readora-title">
            📚 READORA
        </div>

        <div class="readora-subtitle">
            Your Intelligent Book Recommendation Assistant
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 14. WELCOME MESSAGE
# ============================================================

if not st.session_state.messages:

    with st.chat_message(
        "assistant",
        avatar="📚"
    ):

        st.markdown(
            """
            **Welcome to READORA! 👋**

            Tell me what kind of book you're looking for.

            **Examples:**

            > I want a fantasy book involving magic.

            > Recommend a mystery novel involving murder.

            > I want a romantic story about relationships.
            """
        )


# ============================================================
# 15. DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"],
        avatar=(
            "📚"
            if message["role"] == "assistant"
            else "👤"
        )
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# 16. CHAT INPUT
# ============================================================

user_query = st.chat_input(
    "What kind of book are you looking for?"
)


# ============================================================
# 17. HANDLE USER INPUT
# ============================================================

if user_query:

    user_query = user_query.strip()


    # --------------------------------------------------------
    # Save user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )


    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    with st.chat_message(
        "user",
        avatar="👤"
    ):

        st.markdown(
            user_query
        )


    # ========================================================
    # STEP 1: INTENT CLASSIFICATION
    # ========================================================

    with st.chat_message(
        "assistant",
        avatar="📚"
    ):

        with st.spinner(
            "Understanding your request..."
        ):

            try:

                intent = classify_intent(
                    user_query
                )

            except Exception as e:

                print(
                    f"Intent classification failed: {e}"
                )

                # Safe fallback:
                # If intent classification fails,
                # treat it as a book request.
                intent = "book_recommendation"


        print(
            f"Detected intent: {intent}"
        )


        # ====================================================
        # STEP 2: NON-BOOK INTENTS
        # ====================================================

        if intent in {
            "greeting",
            "general_conversation",
            "other"
        }:

            with st.spinner(
                "Preparing a response..."
            ):

                try:

                    answer = (
                        generate_non_retrieval_response(
                            user_query,
                            intent
                        )
                    )

                except Exception as e:

                    print(
                        f"Response generation failed: "
                        f"{e}"
                    )

                    answer = (
                        "Hello! 👋 I'm READORA. "
                        "Tell me what kind of book "
                        "you're looking for."
                    )


            st.markdown(
                answer
            )


        # ====================================================
        # STEP 3: BOOK-RELATED INTENTS
        # ====================================================

        else:

            with st.spinner(
                "Finding the best books for you..."
            ):

                try:

                    result = rag_query(
                        user_query,
                        top_k=5
                    )

                    answer = result.get(
                        "answer",
                        ""
                    )

                    books = result.get(
                        "books",
                        []
                    )


                    # ------------------------------------------------
                    # Final assistant answer
                    # ------------------------------------------------

                    if answer:

                        st.markdown(
                            answer
                        )


                    # ------------------------------------------------
                    # Book cards
                    # ------------------------------------------------

                    if books:

                        st.markdown(
                            "### 📚 Recommended Books"
                        )

                        columns = st.columns(
                            len(books)
                        )


                        for column, book in zip(
                            columns,
                            books
                        ):

                            with column:

                                display_book_card(
                                    book
                                )

                    else:

                        st.info(
                            "I couldn't find "
                            "suitable books."
                        )


                except Exception as e:

                    print(
                        f"RAG Error: {e}"
                    )

                    answer = (
                        "Sorry, I couldn't process "
                        "your request right now."
                    )

                    st.error(
                        f"RAG Error: {e}"
                    )


        # --------------------------------------------------------
        # Save assistant response
        # --------------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )