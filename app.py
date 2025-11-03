import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import gradio as gr

books = pd.read_csv("books_with_emotions.csv")

# Add large_thumbnail field if missing (for safe deployment)
if "large_thumbnail" not in books.columns:
    books["large_thumbnail"] = books["thumbnail"].astype(str) + "&fife=w800"
    books["large_thumbnail"] = books["large_thumbnail"].where(~books["thumbnail"].isna(), "cover-not-found.jpg")

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db_books = Chroma(persist_directory="books_chroma_db", embedding_function=embedding)

categories = ["All"] + sorted(books["simple_categories"].dropna().unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

def retrieve_semantic_recommendations(query, category="All", tone="All", initial_top_k=50, final_top_k=16):
    recs = db_books.similarity_search(query, k=initial_top_k)
    row_indices = [rec.metadata.get("row") for rec in recs if "row" in rec.metadata]
    book_recs = books.iloc[row_indices].drop_duplicates(subset="isbn13")

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    if tone != "All":
        tone_map = {
            "Happy": "joy", "Surprising": "surprise",
            "Angry": "anger", "Suspenseful": "fear", "Sad": "sadness"
        }
        if tone in tone_map:
            book_recs = book_recs.sort_values(by=tone_map[tone], ascending=False)

    return book_recs.head(final_top_k)

def recommend_books(query, category, tone):
    recs = retrieve_semantic_recommendations(query, category, tone)
    results = []
    for _, row in recs.iterrows():
        truncated_description = " ".join(str(row.get("description", "")).split()[:30]) + "..."
        authors_raw = str(row.get("authors", ""))
        authors_split = [a.strip() for a in authors_raw.split(";") if a.strip()]
        if len(authors_split) == 0:
            authors_str = "Unknown Author"
        elif len(authors_split) == 1:
            authors_str = authors_split[0]
        elif len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        else:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"

        caption = f"**{row['title']}** by *{authors_str}*\n\n{truncated_description}"
        results.append((row["large_thumbnail"], caption))
    return results

with gr.Blocks(theme=gr.themes.Glass()) as demo:
    gr.Markdown("# 📚 Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Please enter a description of a book:", placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended books", columns=4, object_fit="cover", height="auto")

    submit_button.click(fn=recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)

if __name__ == "__main__":
    demo.launch()
