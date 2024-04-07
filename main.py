import os
from dotenv import load_dotenv
from supabase import create_client
from PIL import Image
from io import BytesIO
import requests
from sentence_transformers import SentenceTransformer, util
import base64
import streamlit as st
import torch
import numpy as np

# Load environment variables
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

# Create connections to Supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Model configuration
IMAGE_EMBEDDING_MODEL = 'clip-ViT-B-32'
image_embedding_model = SentenceTransformer(IMAGE_EMBEDDING_MODEL)

# Function to create image embedding
# def create_image_embedding(image_path):
#     with open(image_path, "rb") as f:
#         image_data = f.read()
#     image = Image.open(BytesIO(image_data))
#     image_embedding = image_embedding_model.encode([image.convert("RGB")])[0]

#     return base64.b64encode(image_data).decode('utf-8'), image_embedding.tolist()

# # Path to your images folder
# images_folder = "images"

# # Iterate over images in the folder
# for image_file in os.listdir(images_folder):
#     image_path = os.path.join(images_folder, image_file)
#     if os.path.isfile(image_path):
#         image_data, image_embedding = create_image_embedding(image_path)

#         # Upload image and its embedding to Supabase
#         image_row = supabase.table("Image embeddings").insert({
#             "image_data": image_data,
#             "new_image_embeddings": image_embedding
#         }).execute()

#         print(f"Uploaded image: {image_file}")

# print("All images uploaded.")


# Function to create image embedding
def create_image_embedding(image_data):
    image = Image.open(BytesIO(image_data))
    image_embedding = image_embedding_model.encode([image.convert("RGB")])[0]
    return image_data, image_embedding.tolist()

# Function to get embeddings from Supabase
def get_supabase_embeddings():
    response = supabase.table("Image embeddings").select("id", "new_image_embeddings").execute()
    db_embeddings = []
    ids = []
    for record in response.data:
        embedding = record.get('new_image_embeddings')
        if embedding:
            db_embeddings.append(embedding)
            ids.append(record['id'])
    return np.array(db_embeddings, dtype=np.float32), ids

# Function to find similar images
def find_similar_images(query_embedding, db_embeddings, ids, top_k=5):
    query_embedding_tensor = torch.tensor(query_embedding, dtype=torch.float32)
    similarities = util.pytorch_cos_sim(query_embedding_tensor, torch.tensor(db_embeddings))[0].numpy()
    top_k_indices = np.argsort(similarities)[::-1][:top_k]
    return [ids[idx] for idx in top_k_indices]

# Streamlit frontend
st.title("Image Similarity Search")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

    if st.button('Find Similar Images'):
        # Process the uploaded image
        image_data = uploaded_file.read()
        query_image_data, query_embedding = create_image_embedding(image_data)

        # Get embeddings from Supabase
        db_embeddings, ids = get_supabase_embeddings()

        # Find similar images
        similar_ids = find_similar_images(query_embedding, db_embeddings, ids)

        # Display similar images
        for similar_id in similar_ids:
            similar_image_data = supabase.table("Image embeddings").select("image_data").eq("id", similar_id).execute().data[0]['image_data']
            st.image(BytesIO(base64.b64decode(similar_image_data)), caption=f'Similar Image (ID: {similar_id})', use_column_width=True)