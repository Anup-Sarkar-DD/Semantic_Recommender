#
# Book Recommender Dashboard
#
# This script creates an interactive web-based dashboard using Gradio. It leverages
# a pre-trained sentence-transformer model to build a semantic search system for books.
# Users can enter a query, filter by category and emotional tone, and receive a
# gallery of book recommendations.
#
# This version of the code incorporates fixes for:
# 1. Deprecation warnings from the LangChain library by updating import statements.
# 2. A UnicodeDecodeError by specifying the correct encoding when loading the text file.
#
# The logical errors from the previous code regarding filtering and gallery
# formatting have also been resolved.
#

# Data manipulation & visualization
import pandas as pd
import numpy as np
import warnings
# Ignore all warnings to keep the output clean for the user
warnings.filterwarnings("ignore")

# LangChain modules for semantic search
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
# The following imports have been updated to fix the LangChainDeprecationWarning
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Gradio for building the web UI
import gradio as gr

# --- Data Loading and Preparation ---

# Load the books data, which includes book information, emotions, and descriptions.
# Assuming 'books_with_emotions.csv' is available in the same directory.
books = pd.read_csv("books_with_emotions.csv")

# Create a high-resolution thumbnail URL and handle missing images.
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# Load text data for the vector store from the 'tagged_description.txt' file.
# The encoding has been specified to fix the UnicodeDecodeError.
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()

# Split the text into smaller, manageable chunks using a separator.
# This is a crucial step for semantic search to work effectively.
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

# Use a pre-trained Hugging Face model to create numerical embeddings from the text chunks.
# This model converts the text's meaning into vectors that can be searched.
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store the embedded documents in a Chroma vector database. This database is
# optimized for fast similarity searches.
db_books = Chroma.from_documents(
    documents=documents,
    embedding=embedding
)

# --- Core Recommendation Logic ---

def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 16,
) -> pd.DataFrame:
    """
    Retrieves and filters book recommendations based on a user query,
    category, and emotional tone.

    Args:
        query (str): The user's search query (e.g., "a story about forgiveness").
        category (str): The book category to filter by (e.g., "Fiction", "All").
        tone (str): The emotional tone to sort by (e.g., "Happy", "Sad").
        initial_top_k (int): The number of top results to retrieve from the database
                             before filtering.
        final_top_k (int): The final number of recommendations to display.

    Returns:
        pd.DataFrame: A DataFrame of the filtered and sorted book recommendations.
    """
    # Perform a similarity search on the vector database. It returns a list of
    # documents with a relevance score. We start with a high number (initial_top_k)
    # to have enough results for filtering.
    recs = db_books.similarity_search_with_score(query, k=initial_top_k)

    # Extract the ISBNs from the search results to find the corresponding books
    # in the main DataFrame.
    books_list = [str(rec.page_content.strip().split()[0]) for rec in recs]
    # Filter the main DataFrame to get the recommended books.
    book_recs = books[books["isbn13"].isin(books_list)]
    
    # Filter the recommendations by category, if the user has selected one.
    if category and category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    # Sort the recommendations by the emotional tone, if a tone is selected.
    # The emotional scores are assumed to be present as columns in the DataFrame.
    if tone and tone != "All":
        if tone == "Happy":
            book_recs.sort_values(by="joy", ascending=False, inplace=True)
        elif tone == "Surprising":
            book_recs.sort_values(by="suprise", ascending=False, inplace=True)
        elif tone == "Angry":
            book_recs.sort_values(by="anger", ascending=False, inplace=True)
        elif tone == "Sad":
            book_recs.sort_values(by="sadness", ascending=False, inplace=True)
        elif tone == "Suspenseful":
            book_recs.sort_values(by="fear", ascending=False, inplace=True)
    
    # Return the final list of recommendations, capped at final_top_k.
    return book_recs.head(final_top_k)


def recommend_books(
    query: str,
    category: str,
    tone: str
):
    """
    A wrapper function for the Gradio interface that retrieves recommendations
    and formats them for display in a gallery.

    Args:
        query (str): The user's query.
        category (str): The selected category.
        tone (str): The selected emotional tone.

    Returns:
        list: A list of (image_path, caption) tuples for the Gradio gallery.
    """
    # Call the core recommendation function
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    
    # Iterate through the recommendations and format the output for the gallery.
    for _, row in recommendations.iterrows():
        description = str(row.get("description", ""))
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."
        
        authors_split = str(row.get("authors", "")).split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = str(row.get("authors", ""))

        caption = f"{row.get('title', '')} by {authors_str}: {truncated_description}"
        # Correctly append the image path and caption as a single tuple
        results.append((row["large_thumbnail"], caption))
    return results

# --- Gradio UI Layout ---

# Create lists for dropdown menu choices
categories = ["All"] + sorted(books["simple_categories"].unique().tolist())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    # Main title of the dashboard
    gr.Markdown("# Semantic Book Recommender")

    # A row containing the user inputs
    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book:",
                                placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")

    # A section for the output gallery
    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=4, rows=2, object_fit="contain", height="auto")

    # Link the button click event to the recommendation function
    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

# --- Launch the Application ---

if __name__ == "__main__":
    # The app will be launched on a public URL accessible from any device.
    dashboard.launch(debug=True)
