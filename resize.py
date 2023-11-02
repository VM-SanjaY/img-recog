from PIL import Image
import os

# Set the desired size
size = (640, 480)

# Loop through all images in the directory
for filename in os.listdir('path/to/directory'):
    if filename.endswith('.jpg') or filename.endswith('.png'): 
        # Open the image file
        image = Image.open(os.path.join('path/to/directory', filename))
        
        # Resize the image
        resized_image = image.resize(size)
        
        # Save the resized image with a new filename
        new_filename = f"resized_{filename}"
        resized_image.save(os.path.join('path/to/new/directory', new_filename))














