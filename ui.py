import streamlit as st
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from SimplerLLM.tools.generic_loader import load_content
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.embeddings import EmbeddingsLLM, EmbeddingsProvider
from SimplerLLM.tools import text_chunker as chunker

def get_embeddings(texts):
    try:
        embeddings_instance = EmbeddingsLLM.create(provider=EmbeddingsProvider.OPENAI, model_name="text-embedding-3-small")
        batch_size = 10
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            response = embeddings_instance.generate_embeddings(batch_texts)
            embeddings.extend([item.embedding for item in response])
        
        if not embeddings:
            return np.array([])  # Return empty array if no embeddings are generated
        
        return np.array(embeddings)
    except Exception as e:
        st.error(f"An error occurred while generating embeddings: {e}")
        return np.array([])  # Return empty array in case of exception

def compare_with_threshold(input_chunks, target_title, threshold=0.5):
    results = {}
    title_embedding = get_embeddings([target_title])
    chunk_embeddings = get_embeddings(input_chunks)
    
    if chunk_embeddings.size == 0 or title_embedding.size == 0:
        return results  # Return empty results if embeddings are empty

    # Ensure embeddings are 2D
    if len(chunk_embeddings.shape) == 1:
        chunk_embeddings = chunk_embeddings.reshape(1, -1)
    if len(title_embedding.shape) == 1:
        title_embedding = title_embedding.reshape(1, -1)
    
    similarities = cosine_similarity(chunk_embeddings, title_embedding)
    for idx, similarity in enumerate(similarities):
        if similarity[0] >= threshold:
            results[input_chunks[idx]] = similarity[0]
    return results

def choose_the_keyword(target_blog_title, similar_chunks):
    llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4")
    results = {}
    for chunk, _ in similar_chunks:
        prompt = f"""
        You are an expert in SEO and blog interlinking. I have a chunk of text from a blog post which is semantically
        similar to the title of a target blog post, so I can use a keyword from the chunk to link to the blog post.
        Chunk: ```{chunk}```
        Title of target blog post: ```{target_blog_title}```
        The response should be only the keyword and nothing else.
        """
        response = llm_instance.generate_response(prompt=prompt)
        results[chunk] = response
    return results

def process_blog_pair(input_blog_url, target_blog_url):
    try:
        input_blog = load_content(input_blog_url.strip())
        target_blog = load_content(target_blog_url.strip())
    except Exception as e:
        st.error(f"Failed to load content for URLs {input_blog_url} or {target_blog_url}: {e}")
        return []
    
    input_blog_chunks = [chunk.text for chunk in chunker.chunk_by_sentences(input_blog.content).chunk_list]
    target_blog_title = target_blog.title
    similar_chunks = compare_with_threshold(input_blog_chunks, target_blog_title)
    links_data = []

    if similar_chunks:
        keywords = choose_the_keyword(target_blog_title, similar_chunks.items())
        for chunk, keyword in keywords.items():
            link_info = {
                "input_blog": input_blog_url,
                "target_blog": target_blog_url,
                "chunk": chunk,
                "keyword": keyword,
                "cosine_similarity": similar_chunks[chunk]
            }
            links_data.append(link_info)
    return links_data

def process_blogs(blog_urls):
    links_data = []
    with ThreadPoolExecutor() as executor:
        future_to_url = {executor.submit(process_blog_pair, blog_url, target_url): (blog_url, target_url)
                         for blog_url in blog_urls for target_url in blog_urls if blog_url != target_url}
        for future in as_completed(future_to_url):
            links_data.extend(future.result())
    return json.dumps(links_data, indent=4)

st.title("Blog Interlinking Tool")

blog_urls_input = st.text_area("Enter Blog URLs (one per line)")

if st.button("Process Blogs"):
    blog_urls = blog_urls_input.strip().split("\n")
    if len(blog_urls) < 2:
        st.error("Please enter at least two blog URLs.")
    else:
        with st.spinner("Processing..."):
            result = process_blogs(blog_urls)
            st.success("Processing completed!")
            st.json(result)
