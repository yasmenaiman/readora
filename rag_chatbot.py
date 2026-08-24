import os
import time
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES / STREAMLIT SECRETS
# ============================================================

load_dotenv()


def get_secret(name):
    """
    Get a secret from:
    1. Environment variables (local .env)
    2. Streamlit secrets (Streamlit Cloud)
    """

    # Local environment
    value = os.getenv(name)

    if value:
        return value

    # Streamlit Cloud
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

PINECONE_API_KEY = get_secret(
    "PINECONE_API_KEY"
)


if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found. "
        "Add it to Streamlit Secrets."
    )


if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY was not found. "
        "Add it to Streamlit Secrets."
    )

# ============================================================
# 2. CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Embedding model
# ------------------------------------------------------------
# MUST stay the same model used to create the vectors
# already stored in Pinecone.

EMBEDDING_MODEL = "gemini-embedding-2"

EMBEDDING_DIMENSION = 1024


# ------------------------------------------------------------
# Generation model
# ------------------------------------------------------------
# This is separate from the embedding model.

GENERATION_MODEL = "gemini-3.6-flash"


# ------------------------------------------------------------
# Pinecone
# ------------------------------------------------------------

PINECONE_INDEX_NAME = "readora-books"


# ------------------------------------------------------------
# Recommendation configuration
# ------------------------------------------------------------

TOP_K = 5

# We retrieve more candidates than needed so that
# duplicate books/titles can be removed while still
# returning up to TOP_K unique books.
RETRIEVAL_K = 10


# ------------------------------------------------------------
# Full document source
# ------------------------------------------------------------
# document_text is intentionally NOT stored in Pinecone
# metadata according to the READORA document design.
# We retrieve it locally using book_id.

DOCUMENTS_FILE = "data/rag_documents.csv"


# ============================================================
# 3. INITIALIZE CLIENTS
# ============================================================

print("=" * 80)
print("INITIALIZING READORA RAG CHATBOT")
print("=" * 80)

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

pinecone_client = Pinecone(
    api_key=PINECONE_API_KEY
)

index = pinecone_client.Index(
    PINECONE_INDEX_NAME
)

print("Gemini client initialized.")
print("Pinecone client initialized.")
print(
    f"Connected to index: {PINECONE_INDEX_NAME}"
)


# ============================================================
# 4. LOAD RAG DOCUMENT DATA
# ============================================================

print("\n" + "=" * 80)
print("LOADING RAG DOCUMENTS")
print("=" * 80)

documents_df = pd.read_csv(
    DOCUMENTS_FILE
)


# ------------------------------------------------------------
# Validate required columns
# ------------------------------------------------------------

required_document_columns = [
    "book_id",
    "title",
    "document_text",
    "metadata"
]

missing_document_columns = [
    column
    for column in required_document_columns
    if column not in documents_df.columns
]

if missing_document_columns:

    raise ValueError(
        "Missing required columns in rag_documents.csv: "
        f"{missing_document_columns}"
    )


# ------------------------------------------------------------
# Normalize book IDs
# ------------------------------------------------------------

documents_df["book_id"] = pd.to_numeric(
    documents_df["book_id"],
    errors="raise"
).astype(int)


# ------------------------------------------------------------
# Create fast lookup
# ------------------------------------------------------------
# book_id -> complete row

book_lookup = (
    documents_df
    .set_index("book_id")
    .to_dict("index")
)


print(
    f"Documents loaded: "
    f"{len(documents_df):,}"
)


# ============================================================
# 5. CHECK PINECONE INDEX
# ============================================================

print("\n" + "=" * 80)
print("PINECONE INDEX")
print("=" * 80)

stats = index.describe_index_stats()

print(stats)


# ============================================================
# 6. GENERATE QUERY EMBEDDING
# ============================================================

def generate_query_embedding(query):
    """
    Convert the user's query into a vector using
    the same embedding model and dimension used
    when the Pinecone vectors were created.
    """

    result = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=query,
        config={
            "output_dimensionality": EMBEDDING_DIMENSION
        }
    )

    if not result.embeddings:
        raise RuntimeError(
            "Gemini returned no query embedding."
        )

    vector = result.embeddings[0].values

    if len(vector) != EMBEDDING_DIMENSION:

        raise RuntimeError(
            f"Incorrect embedding dimension: "
            f"{len(vector)} instead of "
            f"{EMBEDDING_DIMENSION}"
        )

    return vector


# ============================================================
# 7. RETRIEVE BOOKS FROM PINECONE
# ============================================================

def retrieve_books(
    query,
    top_k=RETRIEVAL_K
):
    """
    Retrieve candidate books from Pinecone.
    """

    print("\nGenerating query embedding...")

    query_vector = generate_query_embedding(
        query
    )

    print(
        f"Query embedding dimension: "
        f"{len(query_vector)}"
    )

    print("\nSearching Pinecone...")

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    matches = results.get(
        "matches",
        []
    )

    return matches


# ============================================================
# 8. ENRICH PINECONE RESULTS
# ============================================================

def enrich_matches(matches):
    """
    Combine Pinecone metadata with the complete document_text
    stored in rag_documents.csv.

    Pinecone provides:
        book_id
        title
        author
        genres
        rating
        year
        image
        score

    rag_documents.csv provides:
        full document_text
    """

    enriched = []

    for match in matches:

        metadata = match.get(
            "metadata",
            {}
        )

        raw_book_id = metadata.get(
            "book_id",
            match.get("id")
        )

        try:

            book_id = int(
                raw_book_id
            )

        except (
            ValueError,
            TypeError
        ):

            print(
                f"Skipping invalid book ID: "
                f"{raw_book_id}"
            )

            continue


        # ----------------------------------------------------
        # Look up complete document
        # ----------------------------------------------------

        book = book_lookup.get(
            book_id
        )

        if book is None:

            print(
                f"Warning: Book ID {book_id} "
                f"exists in Pinecone but was not "
                f"found in rag_documents.csv."
            )

            continue


        # ----------------------------------------------------
        # Build enriched record
        # ----------------------------------------------------

        enriched.append({

            "book_id": book_id,

            "title": metadata.get(
                "title",
                book.get(
                    "title",
                    "Unknown Title"
                )
            ),

            "author_name": metadata.get(
                "author_name",
                "Unknown Author"
            ),

            "genres": metadata.get(
                "genres",
                ""
            ),

            "average_rating": metadata.get(
                "average_rating",
                ""
            ),

            "publication_year": metadata.get(
                "publication_year",
                ""
            ),

            "image_url": metadata.get(
                "image_url",
                ""
            ),

            "similarity_score": match.get(
                "score",
                0.0
            ),

            # Full semantic document
            "document_text": str(
                book.get(
                    "document_text",
                    ""
                )
            )

        })

    return enriched


# ============================================================
# 9. REMOVE DUPLICATE BOOKS
# ============================================================

def deduplicate_books(
    books,
    max_books=TOP_K
):
    """
    Remove duplicate books by:
        1. book_id
        2. normalized title

    This handles cases where Goodreads contains multiple
    records/editions with the same title.
    """

    unique_books = []

    seen_book_ids = set()
    seen_titles = set()

    for book in books:

        book_id = book.get(
            "book_id"
        )

        title = str(
            book.get(
                "title",
                ""
            )
        ).strip().lower()

        # ----------------------------------------------------
        # Duplicate book ID
        # ----------------------------------------------------

        if book_id in seen_book_ids:
            continue


        # ----------------------------------------------------
        # Duplicate title
        # ----------------------------------------------------

        if title and title in seen_titles:
            continue


        seen_book_ids.add(
            book_id
        )

        if title:
            seen_titles.add(
                title
            )

        unique_books.append(
            book
        )


        # ----------------------------------------------------
        # Stop once we have enough unique books
        # ----------------------------------------------------

        if len(unique_books) >= max_books:
            break

    return unique_books


# ============================================================
# 10. BUILD RAG CONTEXT
# ============================================================

def build_context(books):
    """
    Build the textual context sent to the generation model.
    """

    if not books:
        return ""

    context_parts = []

    for rank, book in enumerate(
        books,
        start=1
    ):

        document_text = str(
            book.get(
                "document_text",
                ""
            )
        ).strip()


        if not document_text:

            document_text = (
                "No document text is available "
                "for this book."
            )


        context_parts.append(
            f"""
BOOK {rank}

Book ID:
{book['book_id']}

Title:
{book['title']}

Author:
{book['author_name']}

Genres:
{book['genres']}

Average Rating:
{book['average_rating']}

Publication Year:
{book['publication_year']}

Document:
{document_text}
""".strip()
        )


    return (
        "\n\n"
        + (
            "\n\n"
            + "=" * 70
            + "\n\n"
        ).join(
            context_parts
        )
    )


# ============================================================
# 11. DISPLAY RETRIEVED BOOKS
# ============================================================

def display_results(books):

    print("\n" + "=" * 80)
    print("RETRIEVED BOOKS")
    print("=" * 80)

    if not books:

        print(
            "No books found."
        )

        return


    for rank, book in enumerate(
        books,
        start=1
    ):

        print(
            f"\n{rank}. "
            f"{book['title']}"
        )

        print(
            f"   Book ID: "
            f"{book['book_id']}"
        )

        print(
            f"   Author: "
            f"{book['author_name']}"
        )

        print(
            f"   Genres: "
            f"{book['genres']}"
        )

        print(
            f"   Rating: "
            f"{book['average_rating']}"
        )

        print(
            f"   Publication Year: "
            f"{book['publication_year']}"
        )

        print(
            f"   Similarity Score: "
            f"{book['similarity_score']:.4f}"
        )

        print(
            f"   Image URL: "
            f"{book['image_url']}"
        )

        print(
            "-" * 80
        )


# ============================================================
# 12. GENERATE FINAL RAG ANSWER
# ============================================================

def generate_answer(
    query,
    books
):
    """
    Generate the final natural-language recommendation
    using only the retrieved book context.
    """

    if not books:

        return (
            "I couldn't find relevant books "
            "in the current READORA collection."
        )


    context = build_context(
        books
    )


    prompt = f"""
You are READORA, an intelligent book recommendation assistant.

Your task is to answer the user's request using ONLY
the retrieved book documents below.

USER QUERY:
{query}

RETRIEVED BOOKS:
{context}

RULES:

1. Recommend ONLY books present in the retrieved context.

2. Do not invent:
   - titles
   - authors
   - ratings
   - genres
   - publication years
   - plot details

3. Use the actual document text to explain
   why a book matches the user's request.

4. For each recommendation:
   - mention the title
   - mention the author
   - give a short reason for relevance

5. Do not mention:
   - Pinecone
   - embeddings
   - vector databases
   - similarity scores
   - internal implementation

6. Do not rely only on genre labels when
   the document text provides more useful evidence.

7. Do not recommend the same book twice.

8. Prefer the strongest matches from the
   retrieved books.

9. If the retrieved context is insufficient,
   say so honestly.

10. Keep the answer concise and natural.

Now answer the user's request.
""".strip()


    print(
        f"\nGenerating answer with "
        f"{GENERATION_MODEL}..."
    )


    # --------------------------------------------------------
    # Retry configuration
    # --------------------------------------------------------

    max_retries = 3

    last_error = None


    # ========================================================
    # GENERATION
    # ========================================================

    for attempt in range(
        1,
        max_retries + 1
    ):

        try:

            response = (
                gemini_client
                .models
                .generate_content(
                    model=GENERATION_MODEL,
                    contents=prompt
                )
            )


            if not response.text:

                raise RuntimeError(
                    "Gemini returned an empty response."
                )


            print(
                f"Generation successful with "
                f"{GENERATION_MODEL}"
            )


            return response.text.strip()


        except Exception as e:

            last_error = e

            error_text = str(e)

            print(
                f"Generation error "
                f"(attempt "
                f"{attempt}/{max_retries}):"
            )

            print(
                error_text
            )


            # ------------------------------------------------
            # 503 / unavailable
            # ------------------------------------------------

            if (
                "503" in error_text
                or "UNAVAILABLE" in error_text
            ):

                if attempt < max_retries:

                    wait_seconds = (
                        5 * (
                            2 ** (
                                attempt - 1
                            )
                        )
                    )

                    print(
                        f"Temporary service "
                        f"unavailable. "
                        f"Waiting "
                        f"{wait_seconds} seconds..."
                    )

                    time.sleep(
                        wait_seconds
                    )

                    continue


            # ------------------------------------------------
            # 429 / rate limit
            # ------------------------------------------------

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            ):

                if attempt < max_retries:

                    wait_seconds = (
                        10 * (
                            2 ** (
                                attempt - 1
                            )
                        )
                    )

                    print(
                        f"Rate limit detected. "
                        f"Waiting "
                        f"{wait_seconds} seconds..."
                    )

                    time.sleep(
                        wait_seconds
                    )

                    continue


            # ------------------------------------------------
            # Other errors
            # ------------------------------------------------

            break


    raise RuntimeError(
        "Generation failed after retries.\n"
        f"Last error: {last_error}"
    )


# ============================================================
# 13. COMPLETE RAG PIPELINE
# ============================================================

def rag_query(
    query,
    top_k=TOP_K
):
    """
    Complete READORA RAG pipeline:

    Query
       ↓
    Query Embedding
       ↓
    Pinecone Retrieval
       ↓
    Book ID Lookup
       ↓
    Full document_text
       ↓
    Deduplication
       ↓
    Generation
       ↓
    Final Answer
    """

    query = str(
        query
    ).strip()


    # --------------------------------------------------------
    # Empty query
    # --------------------------------------------------------

    if not query:

        return {
            "query": query,
            "answer": (
                "Please enter a book-related request."
            ),
            "books": []
        }


    print("\n" + "=" * 80)
    print("USER QUERY")
    print("=" * 80)

    print(
        query
    )


    # ========================================================
    # STEP 1: RETRIEVAL
    # ========================================================

    matches = retrieve_books(
        query,
        top_k=max(
            RETRIEVAL_K,
            top_k * 2
        )
    )


    print(
        f"\nRetrieved results from Pinecone: "
        f"{len(matches)}"
    )


    # ========================================================
    # STEP 2: ENRICH WITH DOCUMENT TEXT
    # ========================================================

    books = enrich_matches(
        matches
    )


    print(
        f"Successfully enriched books: "
        f"{len(books)}"
    )


    # ========================================================
    # STEP 3: DEDUPLICATE
    # ========================================================

    books = deduplicate_books(
        books,
        max_books=top_k
    )


    print(
        f"Unique books after deduplication: "
        f"{len(books)}"
    )


    # ========================================================
    # STEP 4: DISPLAY
    # ========================================================

    display_results(
        books
    )


    # ========================================================
    # STEP 5: GENERATION
    # ========================================================

    print("\n" + "=" * 80)
    print("GENERATING RAG ANSWER")
    print("=" * 80)


    answer = generate_answer(
        query,
        books
    )


    print("\n" + "=" * 80)
    print("READORA ANSWER")
    print("=" * 80)

    print(
        answer
    )


    # ========================================================
    # STEP 6: RETURN
    # ========================================================

    return {
        "query": query,
        "answer": answer,
        "books": books
    }


# ============================================================
# 14. LOCAL TEST
# ============================================================

if __name__ == "__main__":

    test_query = (
        "I want a fantasy book involving "
        "magic and supernatural creatures."
    )


    try:

        result = rag_query(
            test_query,
            top_k=TOP_K
        )


    except Exception as e:

        print("\n" + "=" * 80)
        print("RAG ERROR")
        print("=" * 80)

        print(
            e
        )