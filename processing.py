import cv2
import os
import numpy as np
from skimage.feature import hog

def load_data(directory):
    matching_pairs = set()

    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            first_char = filename[0]
            opposite_first_char = 's' if first_char == 'f' else 'f'
            matching_filename = opposite_first_char + filename[1:]
            if matching_filename in os.listdir(directory):
                pair = tuple(sorted((filename, matching_filename)))
                matching_pairs.add(pair)
    
    matching_pairs = list(matching_pairs)
    return matching_pairs

def extract_minutiae(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    binary_image = cv2.medianBlur(binary_image, 3)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minutiae = []

    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)
        hull = cv2.convexHull(simplified_contour, returnPoints=True)
        for point in hull:
            if len(minutiae) < 50:
                minutiae.append(tuple(point[0]))

    while len(minutiae) < 50:
        point_index = np.random.randint(0, len(minutiae))
        point = minutiae[point_index]
        offset = np.random.uniform(-5, 5, size=2)
        new_point = tuple(point[0] + offset)
        minutiae.append(new_point)

    return minutiae

def calculate_minutiae_distances(minutiae1, minutiae2):
    distances = []
    for point1 in minutiae1:
        for point2 in minutiae2:
            distance = np.linalg.norm(np.array(point1) - np.array(point2))
            distances.append(distance)

    return distances

def calculate_hog(image_path):
    # Calculate HOG features from the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    
    # Resize the image to standard size for HOG (optional)
    resized_image = cv2.resize(binary_image, (128, 128))

    # Set up the HOG descriptor and compute HOG features
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(resized_image)
    
    return hog_features.flatten()


def calculate_similarity(image1, image2, method='hog'):
    if method == 'hog':
        # Use HOG similarity (based on feature vectors)
        dist = np.linalg.norm(image1 - image2)  # Euclidean distance between HOG feature vectors
        return dist
    elif method == 'sift':
        # Use SIFT similarity (based on keypoint descriptors)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(image1, image2)
        return sum([m.distance for m in matches])
    else:
        # Default method (Minutiae comparison)
        return np.linalg.norm(np.array(image1) - np.array(image2))