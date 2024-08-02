import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from SimplerLLM.tools.generic_loader import load_content
from SimplerLLM.language.llm import LLM, LLMProvider
from SimplerLLM.language.embeddings import LLM as EmbeddingLLM, EmbeddingsProvider
from SimplerLLM.tools import text_chunker as chunker

def get_embeddings(texts):
    try:
        # Create an instance of EmbeddingLLM 
        embeddings_instance = EmbeddingLLM.create(provider=EmbeddingsProvider.OPENAI, model_name="text-embedding-3-small")
        
        # Generate embeddings for the chunks of sentences
        response = embeddings_instance.generate_embeddings(texts)
        
        # Extract embeddings from the response, convert them to an array, and return them
        embeddings = np.array([item.embedding for item in response])
        return embeddings
    
    # Handle exceptions that might occur during the process
    except Exception as e:
        print("An error occurred:", e)
        return np.array([])

def compare_with_threshold(input_chunks, target_title, threshold=0.5):
    results = {}
    # Get embeddings for the target title and input chunks
    title_embedding = get_embeddings([target_title])
    chunk_embeddings = get_embeddings(input_chunks)
    
    # Compute cosine similarity and filter results
    similarities = cosine_similarity(chunk_embeddings, title_embedding)
    for idx, similarity in enumerate(similarities):
        if similarity[0] >= threshold:
            results[input_chunks[idx]] = similarity[0]
    
    return results

def choose_the_keyword(target_blog_title, similar_chunks):
    # Create an instance of LLM
    llm_instance = LLM.create(provider=LLMProvider.OPENAI, model_name="gpt-4")
    
    # Initialize a dictionary to store the results
    results = {}

    for chunk, _ in similar_chunks:
        prompt =  f"""
        You are an expert in SEO and blog interlinking. I have a chunk of text from a blog post which is semantically
        similar to the title of a target blog post, so I can use a keyword from the chunk to link to the blog post.

        I'm gonna give you both the chunk and the title of the target blog delimited between triple backticks, and 
        your task is to tell me which keyword in the chunk can be used to link to the target blog post. Make sure
        to analyze both of them thoroughly to choose the right keyword from the chunk.

        #Inputs:
        chunk: ```{chunk}```
        title of target blog post: ```{target_blog_title}```

        #Output:
        The keyword can be 2-3 words if necessary and it should be in the chunk. 
        And, the response should be only the keyword and nothing else.
        """ 
        response = llm_instance.generate_response(prompt=prompt)
        results[chunk] = response

    return results
       
# Inputs
input_blog = load_content("https://learnwithhasan.com/ai-paraphraser-tool/")
target_blog = load_content("https://learnwithhasan.com/saas-on-wordpress/")

chunks_model = chunker.chunk_by_sentences(input_blog.content)
chunks_list = chunks_model.chunk_list
input_blog_chunks = [chunk.text for chunk in chunks_list]
print("Chunks Extracted Successfully! \n")
target_blog_title = target_blog.title

# Compare and get results
similar_chunks = compare_with_threshold(input_blog_chunks, target_blog_title)
keywords = choose_the_keyword(target_blog_title, similar_chunks.items())

print("Chunks with high similarity: \n ")
for chunk, cs in similar_chunks.items():
    print("Chunk:", chunk)
    print("Cosine Similarity:", cs)
    print("\n")

print("Keywords in the chunk we can use to link: \n")
for chunk, keyword in keywords.items():
    print("Chunk:", chunk)
    print("Keyword:", keyword)
    print("\n")