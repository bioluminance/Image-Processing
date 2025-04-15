import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Create output directory if it doesn't exist
os.makedirs("output_images", exist_ok=True)

def load_image(path):
    image = Image.open(path)
    return np.array(image)

def display_image(image_array, title="Image", cmap=None):
    plt.imshow(image_array, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()

def save_image(image_array, filename):
    image = Image.fromarray(image_array)
    image.save(f"output_images/{filename}")
    print(f"Saved: output_images/{filename}")

def to_grayscale(image_array):
    grayscale = np.mean(image_array, axis=2).astype(np.uint8)
    return grayscale

def flip_image(image_array, direction='horizontal'):
    if direction == 'horizontal':
        return np.fliplr(image_array)
    elif direction == 'vertical':
        return np.flipud(image_array)
    else:
        raise ValueError("Direction must be 'horizontal' or 'vertical'.")

def rotate_image(image_array, angle=90):
    if angle == 90:
        return np.rot90(image_array)
    elif angle == 180:
        return np.rot90(image_array, 2)
    elif angle == 270:
        return np.rot90(image_array, 3)
    else:
        raise ValueError("Angle must be 90, 180, or 270 degrees.")

def separate_channels(image_array):
    red = image_array[:, :, 0]
    green = image_array[:, :, 1]
    blue = image_array[:, :, 2]
    return red, green, blue

def edge_detection(gray_image):
    # Sobel filter for edge detection
    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    Gx = convolve2d(gray_image, Kx)
    Gy = convolve2d(gray_image, Ky)
    
    G = np.sqrt(Gx**2 + Gy**2)
    G = (G / G.max()) * 255  # normalize
    return G.astype(np.uint8)

def convolve2d(image, kernel):
    height, width = image.shape
    k_height, k_width = kernel.shape
    pad_h = k_height // 2
    pad_w = k_width // 2

    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
    result = np.zeros_like(image)

    for i in range(height):
        for j in range(width):
            region = padded_image[i:i + k_height, j:j + k_width]
            result[i, j] = np.sum(region * kernel)
    return result

def main():
    # Load image
    img_path = "sample_images/image1.jpg"
    img = load_image(img_path)
    display_image(img, "Original Image")

    # Grayscale
    gray = to_grayscale(img)
    display_image(gray, "Grayscale Image", cmap='gray')
    save_image(gray, "grayscale.jpg")

    # Flip and Rotate
    flipped = flip_image(img, 'horizontal')
    display_image(flipped, "Flipped Image")
    save_image(flipped, "flipped.jpg")

    rotated = rotate_image(img, 90)
    display_image(rotated, "Rotated Image")
    save_image(rotated, "rotated.jpg")

    # Channel Separation
    r, g, b = separate_channels(img)
    save_image(r, "red_channel.jpg")
    save_image(g, "green_channel.jpg")
    save_image(b, "blue_channel.jpg")

    # Edge Detection
    edges = edge_detection(gray)
    display_image(edges, "Edge Detection", cmap='gray')
    save_image(edges, "edges.jpg")

if __name__ == "__main__":
    main()
