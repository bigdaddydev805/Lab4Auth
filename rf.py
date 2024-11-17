from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import numpy as np
import os

from processing import load_data, extract_minutiae, calculate_minutiae_distances

def load_and_process_data(directory, data_pairs):
    X_data, y_data = [], []
    pbar = tqdm(total=len(data_pairs), desc=f"Processing {directory} Data...")
    for pair in data_pairs:
        pbar.update(1)
        image1_path = os.path.join(directory, pair[0])
        image2_path = os.path.join(directory, pair[1])
        minutiae1 = extract_minutiae(image1_path)
        minutiae2 = extract_minutiae(image2_path)
        distances = calculate_minutiae_distances(minutiae1, minutiae2)
        X_data.append(distances)
        avg_dist = sum(distances) / len(distances)
        label = 1 if avg_dist < 220 else 0
        y_data.append(label)
    pbar.close()
    return np.array(X_data).reshape(len(X_data), -1), np.array(y_data)

def run_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()
    frr = fn / (fn + tp)
    far = fp / (fp + tn)
    return frr, far

def main():
    training_dir = 'C:/School/CSEC 472/train'
    testing_dir = 'C:/School/CSEC 472/test'
    
    print('Loading Training Data...')
    training_data = load_data(training_dir)
    print('Loading Testing Data...')
    testing_data = load_data(testing_dir)

    X_train, y_train = load_and_process_data(training_dir, training_data)
    X_test, y_test = load_and_process_data(testing_dir, testing_data)

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
