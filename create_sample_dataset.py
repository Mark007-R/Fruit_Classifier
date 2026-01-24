import os
import numpy as np
from PIL import Image, ImageDraw
import random
import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def create_fruit_image(fruit_name, color, size=(200, 200)):
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    margin = 40
    if fruit_name == 'banana':
        points = [
            (margin, size[1]//2),
            (margin+30, margin+20),
            (size[0]-margin-30, margin),
            (size[0]-margin, size[1]//2-20),
            (size[0]-margin-30, size[1]-margin),
            (margin+30, size[1]-margin-20)
        ]
        draw.polygon(points, fill=color, outline='black', width=3)
    elif fruit_name == 'grape':
        radius = 25
        positions = [
            (size[0]//2, size[1]//2-30),
            (size[0]//2-35, size[1]//2),
            (size[0]//2+35, size[1]//2),
            (size[0]//2-20, size[1]//2+35),
            (size[0]//2+20, size[1]//2+35),
        ]
        for x, y in positions:
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                        fill=color, outline='black', width=2)
    else:
        draw.ellipse([margin, margin, size[0]-margin, size[1]-margin], 
                    fill=color, outline='black', width=3)
        if fruit_name == 'apple':
            stem_x = size[0]//2
            draw.rectangle([stem_x-5, margin-20, stem_x+5, margin], 
                         fill='brown', outline='black')
    
    pixels = np.array(img)
    noise = np.random.randint(-15, 15, pixels.shape, dtype=np.int16)
    pixels = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)
    
    return img

def create_dataset():
    fruits = {
        'apple': [(200, 50, 50), (180, 40, 40), (220, 60, 60), (190, 45, 45)],
        'banana': [(255, 220, 50), (250, 210, 40), (255, 230, 60), (245, 205, 45)],
        'orange': [(255, 140, 30), (250, 135, 25), (255, 145, 35), (245, 130, 28)],
        'grape': [(120, 80, 160), (110, 70, 150), (130, 90, 170), (115, 75, 155)]
    }
    dataset_path = 'sample_dataset'
    images_per_class = 25
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        logging.info(f"Created dataset folder: {dataset_path}")
    for fruit_name, colors in fruits.items():
        fruit_folder = os.path.join(dataset_path, fruit_name)
        if not os.path.exists(fruit_folder):
            os.makedirs(fruit_folder)
        logging.info(f"Creating {fruit_name} images...")
        for i in range(images_per_class):
            base_color = random.choice(colors)
            color = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in base_color)
            size = (random.randint(180, 220), random.randint(180, 220))
            img = create_fruit_image(fruit_name, color, size)
            if random.random() > 0.5:
                angle = random.randint(-15, 15)
                img = img.rotate(angle, fillcolor='white')
            img_path = os.path.join(fruit_folder, f"{fruit_name}_{i+1:03d}.jpg")
            img.save(img_path, quality=90)
            if (i + 1) % 5 == 0:
                logging.info(f"  Created {i+1}/{images_per_class} images for {fruit_name}")
        logging.info(f"Completed {fruit_name}: {images_per_class} images")
    logging.info("DATASET CREATION COMPLETE!")
    logging.info(f"Location: {os.path.abspath(dataset_path)}")
    for fruit_name in fruits.keys():
        fruit_folder = os.path.join(dataset_path, fruit_name)
        count = len([f for f in os.listdir(fruit_folder) if f.endswith('.jpg')])
        logging.info(f"  {fruit_name}/  ({count} images)")
    logging.info(f"Total images: {len(fruits) * images_per_class}")

if __name__ == "__main__":
    try:
        create_dataset()
    except Exception as e:
        logging.error(f"Error: {e}")
        logging.error("Make sure you have Pillow installed: pip install Pillow")