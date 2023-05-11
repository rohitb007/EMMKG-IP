import os
from PIL import Image
import imagehash

def similarity(event):
    # Define the threshold for image similarity
    threshold = 10

    # Load the images
    img_folder = "./"+event
    images = []
    for file in os.listdir(img_folder):
        if file.endswith(".jpg"):
            img_path = os.path.join(img_folder, file)
            img = Image.open(img_path)
            images.append((img, img_path))

    # Compute the hash value for each image
    hashes = []
    for img, img_path in images:
        hash_val = imagehash.average_hash(img)
        hashes.append((hash_val, img_path))

    # Compare the hash values of each image
    similar_images = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hashes[i][0] - hashes[j][0] < threshold:
                if hashes[j][1] not in similar_images:
                    similar_images.append(hashes[j][1])

    # Remove the similar images
    for img_path in similar_images:
        # os.remove(img_path)
        print(img_path)
    

similarity(event)
