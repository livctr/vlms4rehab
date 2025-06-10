if __name__ == "__main__":
    from transformers import CLIPProcessor, CLIPModel
    import torch
    from PIL import Image
    import requests

    # Load CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

    # Inputs
    image_paths = [
        "https://images.unsplash.com/photo-1587614203976-365c74645e83",  # dog
        "https://images.unsplash.com/photo-1519681393784-d120267933ba",  # computer
    ]
    texts = [
        "Two women sitting in pink chairs, having a conversation with a laptop on a table between them.",
        "A lone man climbing a snowy mountain.",
        "An aerial view of a bustling city at night.",
        "A close-up of a bowl of fruit on a wooden table."
    ]

    # Load and preprocess images
    images = [Image.open(requests.get(url, stream=True).raw).convert("RGB") for url in image_paths]

    # Process inputs
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

    import pdb ; pdb.set_trace()

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Get embeddings
    image_embeds = outputs.image_embeds  # (num_images, hidden_dim)
    text_embeds = outputs.text_embeds    # (num_texts, hidden_dim)

    # Normalize the embeddings
    image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    # Compute cosine similarity matrix
    similarity_matrix = torch.matmul(image_embeds, text_embeds.T)  # (num_images, num_texts)

    # Print similarity matrix
    print("Similarity matrix (rows: images, columns: texts):")
    print(similarity_matrix)
