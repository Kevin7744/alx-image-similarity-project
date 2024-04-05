import os
from dotenv import load_dotenv
from supabase import create_client
from PIL import Image
from io import BytesIO
import requests
from sentence_transformers import SentenceTransformer
import base64

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
def create_image_embedding(image_path):
    with open(image_path, "rb") as f:
        image_data = f.read()
    image = Image.open(BytesIO(image_data))
    image_embedding = image_embedding_model.encode([image.convert("RGB")])[0]
    return base64.b64encode(image_data).decode('utf-8'), image_embedding.tolist()

# Path to your images folder
images_folder = "images"

# Iterate over images in the folder
for image_file in os.listdir(images_folder):
    image_path = os.path.join(images_folder, image_file)
    if os.path.isfile(image_path):
        image_data, image_embedding = create_image_embedding(image_path)

        # Upload image and its embedding to Supabase
        image_row = supabase.table("Image embeddings").insert({
            "image_data": image_data,
            "new_image_embeddings": image_embedding
        }).execute()

        print(f"Uploaded image: {image_file}")

print("All images uploaded.")



# create frontend

# Take userinput and compare it with embeddings

# display similar images based on the input
