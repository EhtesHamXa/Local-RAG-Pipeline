import fitz  # PyMuPDF
import ollama
import numpy as np

# STEP 1: READ THE PDF
def extract_text_from_pdf(pdf_path):
    print("1. Extracting text from PDF...")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# STEP 2: CHOP THE TEXT INTO OVERLAPPING CHUNKS (V2 UPGRADE!)
def chunk_text(text, chunk_words=200, overlap=50):
    print("2. Chopping text into overlapping chunks...")
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        # Grab 200 words
        chunk_slice = words[i : i + chunk_words]
        chunks.append(" ".join(chunk_slice))
        # Move forward by 150 words (leaving a 50 word overlap!)
        i += (chunk_words - overlap)
    return chunks

# STEP 3: THE SEARCH ENGINE (EMBEDDINGS)
def embed_chunks(chunks):
    print("3. Converting chunks into Embeddings...")
    embeddings = []
    for chunk in chunks:
        response = ollama.embeddings(model='nomic-embed-text', prompt=chunk)
        embeddings.append(response['embedding'])
    return np.array(embeddings)

# STEP 4: TOP-K RETRIEVAL & GENERATION (V2 UPGRADE!)
def chat_with_pdf(question, chunks, embeddings, top_k=3):
    print(f"\nUser Question: {question}")
    
    # Embed the user's question
    question_embedding = ollama.embeddings(model='nomic-embed-text', prompt=question)['embedding']
    
    # MAGIC MATH: Calculate Cosine Similarity for every chunk
    similarities = np.dot(embeddings, question_embedding)
    
    # Get the indices of the Top 3 highest scores
    # argsort sorts lowest to highest, so we take the last 3 [-3:], and reverse them [::-1]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    best_chunks = [chunks[i] for i in top_indices]
    
    print(f"--- 🔍 RETRIEVAL: FOUND THE TOP {top_k} MOST RELEVANT PARAGRAPHS ---")
    # Glue them together with a separator for the AI to read
    context = "\n\n---\n\n".join(best_chunks)
    print(context[:300] + "\n\n... [context truncated for display in terminal] ...")
    print("------------------------------------------------------------------\n")
    
    prompt = f"Read the following context carefully:\n\n{context}\n\nBased ONLY on the context above, answer this question: {question}"
    
    print("4. Asking Phi3 to read the glued context and answer...")
    response = ollama.chat(model='phi3', messages=[
        {'role': 'system', 'content': 'You are a helpful HR assistant. You answer questions strictly using the provided context. If the answer is not in the context, say "I do not know".'},
        {'role': 'user', 'content': prompt}
    ])
    
    print(f"\n--- 🤖 PHI3 FINAL ANSWER ---")
    print(response['message']['content'])


# --- RUN THE PIPELINE! ---

pdf_path = "Employee Handbook - 2026.pdf"

# 1. Read
text = extract_text_from_pdf(pdf_path)
# 2. Chunk (Now using overlapping words!)
chunks = chunk_text(text, chunk_words=200, overlap=50)
# 3. Embed
embeddings = embed_chunks(chunks)

# 4. Chat! 
my_question = "What is the policy about accural bonus? And what if i resign?"
chat_with_pdf(my_question, chunks, embeddings, top_k=3)
