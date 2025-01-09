from numpy import dot
from numpy.linalg import norm

def cosine_similarity(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
        vec1 (list or numpy array): First vector.
        vec2 (list or numpy array): Second vector.
    
    Returns:
        float: Cosine similarity value between -1 and 1.
    """
    if len(vec1) != len(vec2):
        raise ValueError("Both vectors must have the same length.")
    
    # Calculate cosine similarity
    similarity = dot(vec1, vec2) / (norm(vec1) * norm(vec2))
    return similarity

# Input vectors
vec1 = list(map(float, input("Enter the first vector (comma-separated): ").split(',')))
vec2 = list(map(float, input("Enter the second vector (comma-separated): ").split(',')))

try:
    result = cosine_similarity(vec1, vec2)
    print(f"Cosine Similarity: {result}")
except ValueError as e:
    print(e)
