from sentence_transformers import SentenceTransformer

# Model name from Hugging Face
model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Load and download the model
model = SentenceTransformer(model_name)

# Save it in a folder named 'local_embedding_model'
model.save("./local_embedding_model")

print("Model downloaded and saved to ./local_embedding_model")
