from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import numpy as np
import os

from processing import load_data, extract_minutiae, calculate_hog, calculate_similarity

def load_and_process_data(directory, data_pairs, method='hog'):
    X_data, y_data = [], []
    pbar = tqdm(total=len(data_pairs), desc=f"Processing {directory} Data...")
    for pair in data_pairs:
        pbar.update(1)
        image1_path = os.path.join(directory, pair[0])
        image2_path = os.path.join(directory, pair[1])
        
        if method == 'hog':
            # Extract HOG features
            features1 = calculate_hog(image1_path)
            features2 = calculate_hog(image2_path)
            similarity = calculate_similarity(features1, features2, method='hog')

        X_data.append([similarity])
        
        # Create label based on similarity (adjust threshold as needed)
        label = 1 if similarity < 0.8 else 0
        y_data.append(label)
    
    pbar.close()
    return np.array(X_data).reshape(len(X_data), -1), np.array(y_data)

from sklearn.metrics import confusion_matrix

def run_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Ensure confusion matrix uses the correct labels for binary classification
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    
    if cm.size == 1:  # In case we only have one class in the confusion matrix
        # Handle case where only one class is predicted
        print("Warning: Only one class predicted. This can happen if the model is biased or the data is imbalanced.")
        return None, None

    tn, fp, fn, tp = cm.ravel()
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    return frr, far


def main():
    training_dir = 'C:/School/CSEC 472/train'
    testing_dir = 'C:/School/CSEC 472/test'
    
    method = 'hog'  # Change to 'sift' or 'minutiae' to use a different method
    
    print('Loading Training Data...')
    training_data = load_data(training_dir)
    print('Loading Testing Data...')
    testing_data = load_data(testing_dir)

    X_train, y_train = load_and_process_data(training_dir, training_data, method=method)
    X_test, y_test = load_and_process_data(testing_dir, testing_data, method=method)

    print('Modeling...')
    
    frr_values, far_values = [], []
    
    # Perform multiple runs for averaging
    for _ in range(10):
        frr, far = run_model(X_train, y_train, X_test, y_test)
        frr_values.append(frr)
        far_values.append(far)
    
    max_frr = max(frr_values)
    min_frr = min(frr_values)
    avg_frr = sum(frr_values) / len(frr_values)

    max_far = max(far_values)
    min_far = min(far_values)
    avg_far = sum(far_values) / len(far_values)

    print(f"Max FRR: {max_frr}, Min FRR: {min_frr}, Avg FRR: {avg_frr}")
    print(f"Max FAR: {max_far}, Min FAR: {min_far}, Avg FAR: {avg_far}")

    # Optionally, you can also compute and print the Equal Error Rate (EER)
    eer = (avg_frr + avg_far) / 2
    print(f"Equal Error Rate (EER): {eer}")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    main()
