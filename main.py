# imports
import os
from dotenv import load_dotenv
from supabase import create_client
from PIL import Image
from io import BytesIO
import requests
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")


# create connections to supabase
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# model configuration
IMAGE_EMBEDDING_MODEL = 'clip-ViT-B-32'
image_embedding_model = SentenceTransformer(IMAGE_EMBEDDING_MODEL)


# load images


# convert images to embeddings


# create frontend

# Take userinput and compare it with embeddings

# display similar images based on the input
