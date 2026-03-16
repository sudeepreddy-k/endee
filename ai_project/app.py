import streamlit as st
import time
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

# Connect to local Endee server
@st.cache_resource
def get_endee_client():
    client = Endee()
    # Ensure index exists
    try:
        client.get_index("semantic_search")
    except Exception:
        # Create index if it doesn't exist
        client.create_index(
            name="semantic_search",
            dimension=384,
            space_type="cosine",
            precision=Precision.INT8
        )
    return client

@st.cache_resource
def get_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

st.title("Semantic Text Search using Endee Vector Database")

st.markdown("""
This demo application showcases how to build a semantic search tool using the [Endee](https://endee.io) Vector Database.
The system uses the `all-MiniLM-L6-v2` HuggingFace embedding model to convert text snippets into high-dimensional vectors, which are then indexed and searched in Endee.
""")

client = get_endee_client()
index = client.get_index("semantic_search")
model = get_model()

# --- Insert Data Section ---
st.header("1. Insert Knowledge")
snippet = st.text_area("Enter a text snippet to add to the knowledge base:")
if st.button("Add to Database"):
    if snippet.strip():
        with st.spinner("Generating embeddings and inserting into Endee..."):
            vectorId = f"doc_{int(time.time()*1000)}"
            embedding = model.encode(snippet).tolist()
            index.upsert([
                {
                    "id": vectorId,
                    "vector": embedding,
                    "meta": {"text": snippet}
                }
            ])
        st.success("Successfully added to Endee vector representation!")
    else:
        st.warning("Please enter some text.")

st.divider()

# --- Search Section ---
st.header("2. Semantic Search")
query = st.text_input("Enter your search query:")
if st.button("Search"):
    if query.strip():
        with st.spinner("Generating query vector and searching Endee..."):
            query_embedding = model.encode(query).tolist()
            results = index.query(vector=query_embedding, top_k=3)
            
            if results:
                st.subheader("Top Results:")
                for i, res in enumerate(results):
                    similarity = getattr(res, "similarity", getattr(res, "score", 0.0))
                    meta = getattr(res, "meta", getattr(res, "payload", {}))
                    text_match = meta.get("text", "No text provided in meta")
                    st.markdown(f"**Result {i+1} (Score: {similarity:.4f})**")
                    st.info(text_match)
            else:
                st.info("No matching results found in the database.")
    else:
        st.warning("Please enter a search query.")
