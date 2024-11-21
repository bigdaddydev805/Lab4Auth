from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
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


def train_models(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=5)
    rf = RandomForestClassifier(n_estimators=100)
    svc = SVC(kernel='linear', probability=True)

    knn.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    svc.fit(X_train, y_train)

    return knn, rf, svc


def hybrid_predict(knn, rf, svc, X_test):
    knn_preds = knn.predict(X_test)
    rf_preds = rf.predict(X_test)
    svc_preds = svc.predict(X_test)

    # Majority voting
    predictions = []
    for knn_pred, rf_pred, svc_pred in zip(knn_preds, rf_preds, svc_preds):
        votes = [knn_pred, rf_pred, svc_pred]
        final_prediction = max(set(votes), key=votes.count)  # Majority vote
        predictions.append(final_prediction)

    return np.array(predictions)


def evaluate_model(y_test, predictions):
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()
    frr = fn / (fn + tp)
    far = fp / (fp + tn)
    return frr, far


def main():
    training_dir = 'C:/School/CSEC 472/train'
    testing_dir = 'C:/School/CSEC 472/test'

    # Load and process data
    print('Loading Training Data...')
    training_data = load_data(training_dir)
    print('Loading Testing Data...')
    testing_data = load_data(testing_dir)

    X_train, y_train = load_and_process_data(training_dir, training_data)
    X_test, y_test = load_and_process_data(testing_dir, testing_data)

    print('Training Models...')
    knn, rf, svc = train_models(X_train, y_train)

    print('Hybrid Prediction...')
    predictions = hybrid_predict(knn, rf, svc, X_test)

    print('Evaluating Hybrid Model...')
    frr, far = evaluate_model(y_test, predictions)

    print(f"FRR: {frr}")
    print(f"FAR: {far}")

    # Optionally, compute Equal Error Rate (EER)
    eer = (frr + far) / 2
    print(f"Equal Error Rate (EER): {eer}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    main()
