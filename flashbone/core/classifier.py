from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch


# Function to extract features from an image using ResNet
def get_vector(image, model, layer, transform):
    t_img = transform(image).unsqueeze(0)
    my_embedding = torch.zeros(2048)

    def copy_data(m, i, o):
        my_embedding.copy_(o.flatten())

    h = layer.register_forward_hook(copy_data)
    with torch.no_grad():
        model(t_img)
    h.remove()
    return my_embedding


# Function to extract features from masks
def extract_features_from_masks(image, masks, model, layer, transform):
    features = []
    for mask in masks:
        segmentation = mask['segmentation']
        mask_image = np.zeros_like(image)
        mask_image[segmentation] = image[segmentation]
        pil_image = Image.fromarray(mask_image)
        features.append(get_vector(pil_image, model, layer, transform).numpy())
    return np.array(features)


# Function to calculate softmax attention weights
def calculate_attention_weights_softmax(query_embedding, example_embeddings):
    similarities = cosine_similarity(query_embedding.reshape(1, -1), example_embeddings).flatten()
    exp_similarities = np.exp(similarities)
    attention_weights = exp_similarities / np.sum(exp_similarities)
    return attention_weights


# Function to adjust the query embedding
def adjust_embedding(query_embedding, positive_embeddings, negative_embeddings):
    positive_weights = calculate_attention_weights_softmax(query_embedding, positive_embeddings)
    negative_weights = calculate_attention_weights_softmax(query_embedding, negative_embeddings)

    # Compute weighted sums of positive and negative embeddings
    positive_adjustment = np.sum(positive_weights[:, np.newaxis] * positive_embeddings, axis=0)
    negative_adjustment = np.sum(negative_weights[:, np.newaxis] * negative_embeddings, axis=0)

    # Subtract negative adjustment from positive adjustment
    combined_adjustment = positive_adjustment - negative_adjustment
    return combined_adjustment


# Main function to process images
def detect_and_annotate_objects(output_image_path='annotated_output.jpg', num_query_images=5):
    # extract embeddings for classes
    positive_embeddings = [get_vector(img, resnet_model, layer, transform).numpy() for img in query_imgs]
    positive_embeddings = np.array(positive_embeddings, dtype=np.float32)

    negative_embeddings = [get_vector(img, resnet_model, layer, transform).numpy() for img in negative_imgs]
    negative_embeddings = np.array(negative_embeddings, dtype=np.float32)

    # Adjust the query embedding for each query image
    adjusted_query_vectors = np.array([
        adjust_embedding(embedding, positive_embeddings, negative_embeddings)
        for embedding in positive_embeddings
    ])

    # Get mask vectors from the input 
    mask_vectors = np.array(mask_vectors, dtype=np.float32)

    # Normalize query and mask vectors
    adjusted_query_vectors = adjusted_query_vectors / np.linalg.norm(adjusted_query_vectors, axis=1, keepdims=True)
    mask_vectors = mask_vectors / np.linalg.norm(mask_vectors, axis=1, keepdims=True)

    # TODO: adopt fais way, to find matches in multiclass scenario

    # Create FAISS index and add query vectors
    index = faiss.IndexFlatIP(2048)  # Using inner product for cosine similarity
    index.add(adjusted_query_vectors)
    
    # Search for matches in the FAISS index
    similarities, indices = index.search(mask_vectors, 1)

    # Map similarities to [0, 1]
    normalized_similarities = (similarities + 1) / 2

    # Apply a threshold to filter matches
    threshold = 0.474
    filtered_indices = np.where(normalized_similarities > threshold)[0]

    # Draw bounding boxes for detected objects
    for idx in filtered_indices:
        mask = masks[idx]
        segmentation = mask['segmentation']
        coords = np.column_stack(np.where(segmentation))
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
