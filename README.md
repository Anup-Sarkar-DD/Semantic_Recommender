<div style="text-align: center; margin-bottom: 20px;">
  <h1>Semantic Book Recommender (LLMs & Vector Search)</h1>
</div>

- Built a semantic recommender system to suggest books based on similarity of descriptions and emotional tone, using large language models (LLMs).
- Prepared the dataset by:
  - Downloading the 7K Books dataset and cleaning for missing/irrelevant values (title, subtitle, author, category, description, ratings).
  - Filtering out books with descriptions below 25 words for quality recommendations.
  - Standardizing and pairing title/subtitle; tagging book descriptions with unique identifiers.
- Transformed textual features:
  - Converted book descriptions to high-dimensional embeddings using transformer-based models.
  - Generated genre and emotion tags using zero-shot classification and fine-tuned sentiment models.
- Implemented vector search:
  - Created a ChromaDB vector database for fast similarity search across thousands of books.
  - Matched user queries to the most similar book embeddings for personalized recommendations.
- Built interactive dashboard:
  - Developed a user-friendly Gradio interface for querying, viewing results, and filtering by genre/emotion.
  - Visualized semantic matches, similarity scores, and book metadata.
- Evaluated system performance:
  - Measured top-3 recommendation relevancy using real feedback and manual annotation.
  - Ensured robust scaling and low-latency results on large datasets.
- Example results:
  - Users received highly accurate, genre-aware recommendations for queries such as "adventure with a strong female lead" or "uplifting stories with positive emotions."
  - Dashboard enabled interactive exploration and fine-grained control over recommendation preferences.
