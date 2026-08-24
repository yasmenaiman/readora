import os
import json
import html
import base64
from pathlib import Path
from io import BytesIO

import requests
import streamlit as st

from dotenv import load_dotenv
from google import genai
from google.genai import types

from rag_chatbot import rag_query


# ============================================================
# 1. ENVIRONMENT VARIABLES / STREAMLIT SECRETS
# ============================================================

load_dotenv()


def get_secret(name):
    """
    Read a secret from local .env first,
    then from Streamlit Cloud Secrets.
    """

    value = os.getenv(name)

    if value:
        return value

    try:
        value = st.secrets.get(name)

        if value:
            return value

    except Exception:
        pass

    return None


GEMINI_API_KEY = get_secret(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found."
    )


# ============================================================
# 2. CONFIGURATION
# ============================================================

GENERATION_MODEL = "gemini-3.6-flash"

BURGUNDY = "#4A1F2D"
BEIGE = "#E8DCC4"
GOLD = "#C99A2E"
CREAM = "#FAF7F0"
WHITE = "#FFFFFF"
DARK = "#2B2024"
LIGHT_BORDER = "#E6DDD2"


# ============================================================
# 3. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

LOGO_PATH = (
    BASE_DIR
    / "assets"
    / "readora_logo.png"
)


# ============================================================
# 4. GEMINI CLIENT
# ============================================================

intent_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# 5. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="READORA",
    page_icon=str(LOGO_PATH),
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# 6. CUSTOM CSS
# ============================================================

st.markdown(
    f"""
<style>

html,
body,
[data-testid="stAppViewContainer"] {{
    background: {CREAM};
}}

[data-testid="stAppViewContainer"] {{
    background: {CREAM};
}}

[data-testid="stHeader"] {{
    background: transparent;
}}

.block-container {{
    max-width: 900px;
    padding-top: 1rem;
    padding-bottom: 2rem;
}}


/* ============================================================
   HIDE STREAMLIT DEFAULT ELEMENTS
   ============================================================ */

#MainMenu {{
    visibility: hidden;
}}

footer {{
    visibility: hidden;
}}

[data-testid="stToolbar"] {{
    visibility: hidden;
}}


/* ============================================================
   READORA HEADER
   ============================================================ */

.readora-header {{
    background: linear-gradient(
        135deg,
        {BURGUNDY} 0%,
        #5B2838 100%
    );

    border-radius: 22px;

    padding: 1.2rem 1.4rem;

    margin-bottom: 1rem;

    box-shadow:
        0 8px 24px
        rgba(74, 31, 45, 0.18);
}}

.readora-brand {{
    display: flex;

    align-items: center;

    gap: 1rem;
}}

.readora-logo {{
    width: 78px;
    height: 78px;

    object-fit: contain;

    flex-shrink: 0;

    border-radius: 16px;

    background: rgba(255,255,255,0.08);

    padding: 6px;
}}

.readora-logo-fallback {{
    width: 78px;
    height: 78px;

    display: flex;

    align-items: center;
    justify-content: center;

    flex-shrink: 0;

    border-radius: 16px;

    background: {GOLD};

    font-size: 2.2rem;
}}

.readora-brand-text {{
    display: flex;

    flex-direction: column;

    justify-content: center;
}}

.readora-title {{
    color: {WHITE};

    font-size: 2rem;

    font-weight: 800;

    line-height: 1.1;

    letter-spacing: 0.02em;
}}

.readora-tagline {{
    color: rgba(255,255,255,0.82);

    font-size: 0.88rem;

    margin-top: 0.35rem;
}}


/* ============================================================
   WELCOME CARD
   ============================================================ */

.welcome-card {{
    background: {WHITE};

    border: 1px solid {LIGHT_BORDER};

    border-left: 4px solid {GOLD};

    border-radius: 16px;

    padding: 1rem 1.15rem;

    margin-bottom: 1rem;

    box-shadow:
        0 4px 16px
        rgba(74, 31, 45, 0.06);
}}

.welcome-title {{
    color: {BURGUNDY};

    font-size: 1.08rem;

    font-weight: 750;

    margin-bottom: 0.3rem;
}}

.welcome-text {{
    color: {DARK};

    font-size: 0.9rem;

    line-height: 1.55;
}}


/* ============================================================
   CHAT
   ============================================================ */

[data-testid="stChatMessage"] {{
    background: transparent;

    padding-top: 0.35rem;
    padding-bottom: 0.35rem;
}}

[data-testid="stChatMessageContent"] {{
    border-radius: 18px;
}}


/* ============================================================
   BOOK INFO
   ============================================================ */

.book-info {{
    padding-top: 0.35rem;
}}

.book-title {{
    color: {BURGUNDY};

    font-size: 1rem;

    font-weight: 750;

    line-height: 1.35;

    margin-bottom: 0.18rem;
}}

.book-author {{
    color: #76666B;

    font-size: 0.86rem;

    margin-bottom: 0.35rem;
}}

.book-rating {{
    color: {GOLD};

    font-size: 0.88rem;

    font-weight: 700;

    margin-bottom: 0.3rem;
}}

.book-genres {{
    color: #6B5C61;

    font-size: 0.75rem;

    line-height: 1.45;
}}


/* ============================================================
   RECOMMENDATION TITLE
   ============================================================ */

.recommendation-title {{
    color: {BURGUNDY};

    font-size: 1.15rem;

    font-weight: 750;

    margin-top: 1rem;

    margin-bottom: 0.75rem;

    display: flex;

    align-items: center;

    gap: 0.4rem;
}}


/* ============================================================
   CHAT INPUT
   ============================================================ */

[data-testid="stChatInput"] {{
    border: 1px solid {LIGHT_BORDER};

    border-radius: 18px;

    background: {WHITE};
}}

[data-testid="stChatInput"] textarea {{
    color: {DARK};
}}

[data-testid="stChatInput"] button {{
    background: {BURGUNDY};

    border-radius: 12px;
}}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# 7. SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# 8. INTENT CLASSIFICATION
# ============================================================

def classify_intent(user_message):

    prompt = f"""
You are the intent router for READORA,
an intelligent book recommendation assistant.

Classify the user's message into EXACTLY ONE intent.

1. greeting
The user is greeting, welcoming, thanking,
or starting a conversation without asking
for books.

2. book_recommendation
The user wants book recommendations,
suggestions, discovery, or help choosing
what to read.

3. book_information
The user asks about a specific book,
author, genre, plot, rating, publication,
or other book-related information.

4. general_conversation
The user is talking casually without
requesting book retrieval.

5. other
Anything else.

Examples:

"hello" -> greeting
"hi" -> greeting
"اهلا" -> greeting
"أهلاً" -> greeting
"مرحبا" -> greeting
"السلام عليكم" -> greeting
"شكرا" -> greeting

"I want a fantasy book about magic"
-> book_recommendation

"Recommend a mystery novel"
-> book_recommendation

"I need a romantic story"
-> book_recommendation

"What is The Changeling Sea about?"
-> book_information

"Who wrote Harry Potter?"
-> book_information

"How are you?"
-> general_conversation

Return ONLY JSON:

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

    except Exception:
        pass

    return "other"


# ============================================================
# 9. NON-RETRIEVAL RESPONSE
# ============================================================

def generate_non_retrieval_response(
    user_message,
    intent
):

    if intent == "greeting":

        prompt = f"""
You are READORA, a warm and elegant book
recommendation assistant.

The user is simply greeting you.

Reply briefly and naturally.
Do NOT recommend books yet.

Invite the user to tell you what they
would like to read.

User:
{user_message}
"""

    elif intent == "general_conversation":

        prompt = f"""
You are READORA, a friendly book assistant.

Respond naturally to the user's casual
message.

Do not recommend books unless the user
asks for them.

User:
{user_message}
"""

    else:

        prompt = f"""
You are READORA.

The user has not provided enough information
for a useful book search.

Reply briefly and naturally.

User:
{user_message}
"""

    response = intent_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt
    )

    if response.text:
        return response.text.strip()

    return (
        "Welcome to READORA! 📚 "
        "Tell me what kind of book you're looking for."
    )


# ============================================================
# 10. IMAGE URL CLEANING
# ============================================================

def clean_image_url(url):

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
# 11. DOWNLOAD IMAGE
# ============================================================

@st.cache_data(
    show_spinner=False,
    ttl=86400
)
def load_book_image(image_url):

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
                )
            }
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        if not content_type.startswith(
            "image/"
        ):
            return None

        return response.content

    except Exception as e:

        print(
            f"Image loading failed: {image_url}"
        )

        print(
            f"Reason: {e}"
        )

        return None


# ============================================================
# 12. DISPLAY IMAGE
# ============================================================

def display_book_image(image_url):

    image_data = load_book_image(
        image_url
    )

    if image_data:

        try:

            st.image(
                BytesIO(image_data),
                use_container_width=True
            )

            return

        except Exception as e:

            print(
                f"Image display error: {e}"
            )

    st.markdown(
        f"""
        <div style="
            height:260px;
            display:flex;
            align-items:center;
            justify-content:center;

            background:{BEIGE};

            border-radius:16px;

            border:1px solid {LIGHT_BORDER};

            color:{BURGUNDY};

            font-size:2.8rem;
        ">
            📖
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 13. DISPLAY BOOK CARD
# ============================================================

def display_book_card(book):

    title = html.escape(
        str(
            book.get(
                "title",
                "Unknown Title"
            )
        )
    )

    author = html.escape(
        str(
            book.get(
                "author_name",
                "Unknown Author"
            )
        )
    )

    genres = html.escape(
        str(
            book.get(
                "genres",
                ""
            )
            or ""
        )
    )

    rating = book.get(
        "average_rating",
        ""
    )

    image_url = clean_image_url(
        book.get(
            "image_url",
            ""
        )
    )

    display_book_image(
        image_url
    )

    st.markdown(
        f"""
        <div class="book-info">

            <div class="book-title">
                {title}
            </div>

            <div class="book-author">
                by {author}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

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

    if genres:

        st.markdown(
            f"""
            <div class="book-genres">
                {genres}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# 14. READORA HEADER
# ============================================================

def get_logo_base64():

    if not LOGO_PATH.exists():
        return None

    try:

        with open(
            LOGO_PATH,
            "rb"
        ) as f:

            return base64.b64encode(
                f.read()
            ).decode("utf-8")

    except Exception as e:

        print(
            f"Logo loading failed: {e}"
        )

        return None


logo_base64 = get_logo_base64()


if logo_base64:

    st.html(
        f"""
        <div class="readora-header">

            <div class="readora-brand">

                <img
                    src="data:image/png;base64,{logo_base64}"
                    class="readora-logo"
                    alt="READORA logo"
                >

                <div class="readora-brand-text">

                    <div class="readora-title">
                        READORA
                    </div>

                    <div class="readora-tagline">
                        Your Intelligent Book Assistant
                    </div>

                </div>

            </div>

        </div>
        """
    )

else:

    st.html(
        f"""
        <div class="readora-header">

            <div class="readora-brand">

                <div class="readora-logo-fallback">
                    📚
                </div>

                <div class="readora-brand-text">

                    <div class="readora-title">
                        READORA
                    </div>

                    <div class="readora-tagline">
                        Your Intelligent Book Assistant
                    </div>

                </div>

            </div>

        </div>
        """
    )


# ============================================================
# 15. WELCOME
# ============================================================

if not st.session_state.messages:

    st.html(
        f"""
        <div class="welcome-card">

            <div class="welcome-title">
                Welcome to READORA 👋
            </div>

            <div class="welcome-text">
                Tell me what kind of book you're looking for,
                and I'll help you discover something you'll love.
            </div>

        </div>
        """
    )


# ============================================================
# 16. CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    avatar = (
        "📚"
        if message["role"] == "assistant"
        else "👤"
    )

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# 17. CHAT INPUT
# ============================================================

user_query = st.chat_input(
    "Ask READORA what you'd like to read..."
)


# ============================================================
# 18. HANDLE USER MESSAGE
# ============================================================

if user_query:

    user_query = user_query.strip()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    with st.chat_message(
        "user",
        avatar="👤"
    ):

        st.markdown(
            user_query
        )

    with st.chat_message(
        "assistant",
        avatar="📚"
    ):

        try:

            with st.spinner(
                "Understanding your request..."
            ):

                intent = classify_intent(
                    user_query
                )

            print(
                f"Detected intent: {intent}"
            )


            # =================================================
            # NON-BOOK MESSAGE
            # =================================================

            if intent in {
                "greeting",
                "general_conversation",
                "other"
            }:

                with st.spinner(
                    "Preparing your response..."
                ):

                    answer = (
                        generate_non_retrieval_response(
                            user_query,
                            intent
                        )
                    )

                st.markdown(
                    answer
                )


            # =================================================
            # BOOK REQUEST
            # =================================================

            else:

                with st.spinner(
                    "Finding books for you..."
                ):

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

                if answer:

                    st.markdown(
                        answer
                    )

                if books:

                    st.markdown(
                        """
                        <div class="recommendation-title">
                            📚 Recommended Books
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    # Three books per row
                    for start in range(
                        0,
                        len(books),
                        3
                    ):

                        row_books = books[
                            start:start + 3
                        ]

                        columns = st.columns(
                            len(row_books)
                        )

                        for column, book in zip(
                            columns,
                            row_books
                        ):

                            with column:

                                display_book_card(
                                    book
                                )

                else:

                    st.info(
                        "I couldn't find suitable "
                        "books for this request."
                    )


        except Exception as e:

            print(
                f"Application error: {e}"
            )

            answer = (
                "Sorry, I couldn't process "
                "your request right now."
            )

            st.error(
                "Something went wrong while "
                "processing your request."
            )


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )