import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import feature, filters, color
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from skimage.feature import hog
from scipy.fftpack import fft2, fftshift
from typing import List, Tuple
import pandas as pd


def load_images(df: pd.DataFrame, image_size: Tuple[int, int] = None) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load images and labels from a dataframe with image paths and labels.

    Args:
        df (pd.DataFrame): DataFrame containing 'image_path' and 'label' columns.
        image_size (Tuple[int, int]): Desired dimensions to resize each image (width, height).

    Returns:
        Tuple[List[np.ndarray], List[str]]: List of loaded images and their labels.
    """
    images, labels = [], []
    for _, row in df.iterrows():
        curr_image = Image.open(row['image_path'])
        if image_size is not None:
            curr_image = curr_image.resize(image_size)
        images.append(np.array(curr_image))
        labels.append(row['label'])
    return images, labels


def plot_class_distribution(labels: List[str]) -> None:
    """
    Plot the distribution of image classes.

    Args:
        labels (List[str]): List of class labels for the images.
    """
    plt.figure(figsize=(10, 5))
    plt.hist(labels, bins=len(set(labels)), color='skyblue')
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.show()


def plot_color_distribution(images: List[np.ndarray], n_images: int) -> None:
    """
    Plot color distributions for RGB channels across selected images.

    Args:
        images (List[np.ndarray]): List of images as numpy arrays.
        n_images (int): Number of images to analyze.
    """
    all_pixels = np.vstack([img.reshape(-1, 3) for img in images[:n_images]])
    plt.figure(figsize=(12, 6))
    for i, color in enumerate(['Red', 'Green', 'Blue']):
        plt.hist(all_pixels[:, i], bins=50, color=color.lower(), alpha=0.5, label=f"{color} Channel")
    plt.title("Color Channel Distribution")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


def plot_dominant_colors(image: np.ndarray, n_colors: int) -> None:
    """
    Display dominant colors in an image using k-means clustering.

    Args:
        image (np.ndarray): An image represented as a numpy array.
        n_colors (int): Number of dominant colors to find.
    """
    kmeans = KMeans(n_clusters=n_colors)
    img_reshaped = image.reshape(-1, 3)
    kmeans.fit(img_reshaped)
    dominant_colors = kmeans.cluster_centers_.astype(int)

    plt.figure(figsize=(12, 2))
    plt.title(f"Top {n_colors} Dominant Colors")
    for i, color in enumerate(dominant_colors):
        plt.bar(i, 1, color=color / 255, width=1)
    plt.axis('off')
    plt.show()


def plot_edge_detection(images: List[np.ndarray], n_images: int) -> None:
    """
    Apply and display edge detection using the Sobel filter on sample images.

    Args:
        images (List[np.ndarray]): List of images as numpy arrays.
        n_images (int): Number of images to analyze.
    """
    edge_imgs = [filters.sobel(color.rgb2gray(img)) for img in images[:n_images]]
    fig, axs = plt.subplots(1, n_images, figsize=(3 * n_images, 3))
    for i, edge_img in enumerate(edge_imgs):
        axs[i].imshow(edge_img, cmap='gray')
        axs[i].axis('off')
    plt.suptitle("Edge Detection (Sobel Filter)")
    plt.show()


def plot_edge_orientation_distribution(images: List[np.ndarray], n_images: int) -> None:
    """
    Plot the distribution of edge orientations in images.

    Args:
        images (List[np.ndarray]): List of images as numpy arrays.
        n_images (int): Number of images to analyze.
    """
    orientations = []
    for img in images[:n_images]:
        edges = feature.canny(color.rgb2gray(img))
        orientation = np.arctan2(*np.gradient(edges.astype(float)))
        orientations.extend(orientation.ravel())
    plt.figure(figsize=(10, 5))
    plt.hist(orientations, bins=50, color='purple')
    plt.title("Edge Orientation Distribution")
    plt.xlabel("Orientation (radians)")
    plt.ylabel("Frequency")
    plt.show()


def plot_texture_analysis(images: List[np.ndarray], n_images: int) -> None:
    """
    Display texture analysis using Local Binary Patterns (LBP) on sample images.

    Args:
        images (List[np.ndarray]): List of images as numpy arrays.
        n_images (int): Number of images to analyze.
    """
    lbp_imgs = [feature.local_binary_pattern(color.rgb2gray(img), P=8, R=1) for img in images[:n_images]]
    fig, axs = plt.subplots(1, n_images, figsize=(3 * n_images, 3))
    for i, lbp_img in enumerate(lbp_imgs):
        axs[i].imshow(lbp_img, cmap='gray')
        axs[i].axis('off')
    plt.suptitle("Texture Analysis (Local Binary Patterns)")
    plt.show()


def plot_hog_analysis(images: List[np.ndarray], n_images: int) -> None:
    """
    Display Histogram of Oriented Gradients (HOG) for shape detection on sample images.

    Args:
        images (List[np.ndarray]): List of images as numpy arrays.
        n_images (int): Number of images to analyze.
    """
    fig, axs = plt.subplots(1, n_images, figsize=(3 * n_images, 3))
    for i, img in enumerate(images[:n_images]):
        gray_img = color.rgb2gray(img)
        hog_img = hog(gray_img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)[1]
        axs[i].imshow(hog_img, cmap='gray')
        axs[i].axis('off')
    plt.suptitle("Histogram of Oriented Gradients (HOG)")
    plt.show()


def plot_frequency_analysis(images: List[np.ndarray], n_images: int) -> None:
    """
    Display frequency analysis using Fourier Transform on sample images.

    Args:
        images (List[np.ndarray]): List of images as numpy arrays.
        n_images (int): Number of images to analyze.
    """
    fig, axs = plt.subplots(1, n_images, figsize=(3 * n_images, 3))
    for i, img in enumerate(images[:n_images]):
        gray_img = color.rgb2gray(img)
        f_transform = fftshift(fft2(gray_img))
        magnitude_spectrum = np.log(np.abs(f_transform) + 1)
        axs[i].imshow(magnitude_spectrum, cmap='gray')
        axs[i].axis('off')
    plt.suptitle("Frequency Analysis (Fourier Transform)")
    plt.show()


def plot_dimensionality_reduction(images: List[np.ndarray], labels: List[str], n_images: int) -> None:
    """
    Perform PCA on HOG features and display the result.

    Args:
        images (List[np.ndarray]): List of images as numpy arrays.
        labels (List[str]): List of class labels for the images.
        n_images (int): Number of images to analyze.
    """
    hog_features = [hog(color.rgb2gray(img), pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
                    for img in images[:n_images]]
    pca = PCA(n_components=2)
    hog_pca = pca.fit_transform(hog_features)
    plt.figure(figsize=(10, 5))
    plt.scatter(hog_pca[:, 0], hog_pca[:, 1], c=[hash(label) % 10 for label in labels[:n_images]], cmap='viridis')
    plt.title("PCA on HOG Features")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar()
    plt.show()


def plot_outlier_detection(images: List[np.ndarray], n_images: int) -> None:
    """
    Plot distribution of image intensities for outlier detection.

    Args:
        images (List[np.ndarray]): List of images as numpy arrays.
        n_images (int): Number of images to analyze.
    """
    mean_intensities = [np.mean(color.rgb2gray(img)) for img in images[:n_images]]
    plt.figure(figsize=(10, 5))
    plt.hist(mean_intensities, bins=50, color='red')
    plt.title("Image Intensity Distribution")
    plt.xlabel("Mean Intensity")
    plt.ylabel("Frequency")
    plt.show()


# Sample usage:
# images, labels = load_images_from_df(df, (64, 64))
# plot_class_distribution(labels)
# plot_color_distribution(images, n_images=5)
# plot_dominant_colors(images[0], n_colors=5)
# plot_edge_detection(images, n_images=5)
# plot_edge_orientation_distribution(images, n_images=5)
# plot_texture_analysis(images, n_images=5)
# plot_hog_analysis(images, n_images=5)
# plot_frequency_analysis(images, n_images=5)
# plot_dimensionality_reduction(images, labels, n_images=5)
# plot_outlier_detection(images, n_images=5)
