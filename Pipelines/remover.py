import os
import shutil

# Path to the WikiArt dataset
data_dir = 'Wikiart/dataset'


# List of the 13 most popular genres to keep
popular_genres = [
    'Abstract_Expressionism',
    'Baroque',
    'Cubism',
    'Expressionism',
    'Impressionism',
    'Pop Art',
    'Realism',
    'Rococo',
    'Romanticism',
    'Surrealism',
    'Symbolism',
    'Minimalism',
    'Renaissance'
]

# Normalize genre names for comparison (if needed)
popular_genres = [genre.replace(" ", "_") for genre in popular_genres]

# Step through each folder in the dataset
for genre in os.listdir(data_dir):
    genre_path = os.path.join(data_dir, genre)

    # Check if the current folder is a directory and if it's not in the popular genres list
    if os.path.isdir(genre_path) and genre not in popular_genres:
        print(f"Removing folder: {genre_path}")
        shutil.rmtree(genre_path)  # Delete the folder and its contents
