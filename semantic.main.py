import os
os.environ ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
import pandas as pd
import argparse
import faiss

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="semantic_model", choices=["semantic_model", "ckipbert"],
                   help="Type of model to use: 'semantic_model' or 'ckipbert'")
parser.add_argument("--output_dir", type=str, default="search_results", help="Directory to save search results")
parser.add_argument("--query_file", type=str, help="Path to the TXT file containing queries")
args = parser.parse_args()

# Prepare the model and inference function based on the model type
if args.model_type == "semantic_model":
   from semantic_model import get_semantic_model, inference
   model, tokenizer = get_semantic_model()

# Set the embeddings directory based on model type
embeddings_dir = f'./embeddings/{args.model_type}/'

# Load pre-computed product names and embeddings
product_names = []
product_embeddings = []

# Ensure the embeddings directory exists
if not os.path.exists(embeddings_dir):
   raise FileNotFoundError(f"Embeddings directory '{embeddings_dir}' not found.")

# Loop through all .npy files in the embeddings directory
for file in os.listdir(embeddings_dir):
   if file.endswith('.npy'):
       embedding_file = os.path.join(embeddings_dir, file)
       csv_file = os.path.join('./random_samples_1M', file.replace('.npy', '.csv'))

       # Check if the corresponding CSV file exists
       if not os.path.exists(csv_file):
           continue

       # Load product names from the CSV file
       items_df = pd.read_csv(csv_file)
       product_names.extend(items_df['product_name'].values)

       # Load product embeddings from the .npy file
       embeddings = np.load(embedding_file)
       product_embeddings.append(embeddings)

# Concatenate all embeddings into a single numpy array
product_embeddings = np.concatenate(product_embeddings, axis=0)

print(f'Number of products: {len(product_names)}')
print(f'Number of pre-computed embeddings: {product_embeddings.shape[0]}')

# Convert embeddings to float32
product_embeddings = product_embeddings.astype('float32')

# Normalize embeddings for cosine similarity
faiss.normalize_L2(product_embeddings)

# Build FAISS index
embedding_dim = product_embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)  # Using Inner Product as similarity measure
index.add(product_embeddings)

print(f'FAISS index built with {index.ntotal} vectors.')

# Convert product names to pandas Series for easy indexing
product_names_series = pd.Series(product_names)

# Function to search for the top k items (top_k is set to 10)
def search(query, product_names_series, index, top_k=10):
   # Get the embedding for the query
   query_embedding, _ = inference(tokenizer, model, [query], 16)
   
   # Ensure query_embedding is a 1D array and convert it to a 2D array
   query_embedding = np.array(query_embedding).astype('float32')
   
   # Check if the embedding is one-dimensional and reshape if necessary
   if query_embedding.ndim == 1:
       query_embedding = query_embedding.reshape(1, -1)  # Reshape to (1, d)

   # Normalize query embedding
   faiss.normalize_L2(query_embedding)

   # Search using the index
   scores, indices = index.search(query_embedding, top_k)

   # Retrieve search results
   top_k_names = product_names_series.iloc[indices[0]].values
   top_k_scores = scores[0]

   return top_k_names, top_k_scores

# Initialize a list to store queries
queries = []

if args.query_file:
   try:
       with open(args.query_file, 'r', encoding='utf-8') as f:  # 使用 UTF-8 編碼
           queries = [line.strip() for line in f if line.strip()]  # 讀取每行的查詢內容
   except FileNotFoundError:
       print(f"Error: File '{args.query_file}' not found.")
   except UnicodeDecodeError:
       print(f"Error: The file is not properly encoded in UTF-8.")
   except Exception as e:
       print(f"An error occurred: {e}")

# Create a list to store all results for merging
all_results = []

# Proceed with the search using the queries
for query in queries:
   start_time = time.time()
   top_k_names, scores = search(query, product_names_series, index)
   elapsed_time = time.time() - start_time
   print(f'Took {elapsed_time:.4f} seconds to search for query: "{query}" with top_k=10')

   # Append results to the all_results list
   query_results = pd.DataFrame({
       'Query': [query] * len(top_k_names),
       'Rank': list(range(1, len(top_k_names) + 1)),
       'Product Name': top_k_names,
       'Score': scores
   })
   all_results.append(query_results)

# Merge all results into a single DataFrame
merged_results = pd.concat(all_results, ignore_index=True)

# Save the merged results to a single CSV file
output_file = os.path.join(args.output_dir, "semantic_results_250.csv")
os.makedirs(args.output_dir, exist_ok=True)
merged_results.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f'All results saved to {output_file}')
