import imageio
import os
import re

# Directory containing the PNGs
image_dir = '/home/samy/epita/ing2/ppc/portfolio-optimization/generation_plots/'

# List all PNG files in the directory
images = [img for img in os.listdir(image_dir) if img.endswith(".png")]

# Extract numbers from filenames and sort accordingly
def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else -1  # fallback in case no number is found

images.sort(key=extract_number)

# Create the full paths
image_paths = [os.path.join(image_dir, img) for img in images]

# Output path for the GIF
gif_path = '/home/samy/epita/ing2/ppc/portfolio-optimization/generation_plots/animation.gif'

# Create the GIF
with imageio.get_writer(gif_path, mode='I', duration=250, loop=0) as writer:
    for image_path in image_paths:
        image = imageio.imread(image_path)
        writer.append_data(image)

print(f"GIF saved to {gif_path}")
