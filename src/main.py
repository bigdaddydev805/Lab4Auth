import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fingerprint_processor import FingerprintProcessor
from hybrid_matcher import HybridFingerprintMatcher

def load_image_pairs(directory):
    """Load image pairs from directory"""
    f_images = []
    s_images = []
    
    # Load all images
    for filename in sorted(os.listdir(directory)):
        if filename.startswith('f'):
            f_images.append(os.path.join(directory, filename))
        elif filename.startswith('s'):
            s_images.append(os.path.join(directory, filename))
    
    # Sort to ensure matching pairs
    f_images.sort()
    s_images.sort()
    
    return list(zip(f_images, s_images))

def process_dataset(processor, hybrid, dataset_type):
    """Process either training or test dataset"""
    print(f"\nProcessing {dataset_type} dataset...")
    
    results = {
        'method1': {'genuine': [], 'impostor': []},
        'method2': {'genuine': [], 'impostor': []},
        'method3': {'genuine': [], 'impostor': []},
        'hybrid': {'genuine': [], 'impostor': []}
    }
    
    # Load image pairs
    image_pairs = load_image_pairs(f'data/{dataset_type}')
    total_pairs = len(image_pairs)
    
    print(f"Found {total_pairs} image pairs")
    
    # Process genuine pairs
    for idx, (f_path, s_path) in enumerate(image_pairs):
        print(f"Processing pair {idx+1}/{total_pairs}", end='\r')
        
        # Read images
        img1 = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(s_path, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            print(f"Warning: Could not read images: {f_path}, {s_path}")
            continue
            
        # Process with individual methods
        for method_num in range(1, 4):
            method = f'method{method_num}'
            score = processor.process_pair(img1, img2, method)
            results[method]['genuine'].append(score)
            
        # Process with hybrid system
        hybrid_score = hybrid.match_fingerprints(img1, img2)
        results['hybrid']['genuine'].append(hybrid_score)
        
        # Create impostor pairs (match with next image)
        if idx < len(image_pairs) - 1:
            img3 = cv2.imread(image_pairs[idx + 1][0], cv2.IMREAD_GRAYSCALE)
            if img3 is not None:
                for method_num in range(1, 4):
                    method = f'method{method_num}'
                    score = processor.process_pair(img1, img3, method)
                    results[method]['impostor'].append(score)
                    
                hybrid_score = hybrid.match_fingerprints(img1, img3)
                results['hybrid']['impostor'].append(hybrid_score)
    
    print("\nProcessing complete!")
    return results

def calculate_error_rates(genuine_scores, impostor_scores):
    """Calculate FRR, FAR, and EER"""
    thresholds = np.linspace(0, 1, 100)
    frrs = []
    fars = []
    
    for threshold in thresholds:
        frr = np.mean(np.array(genuine_scores) < threshold)
        far = np.mean(np.array(impostor_scores) >= threshold)
        
        frrs.append(frr)
        fars.append(far)
    
    frrs = np.array(frrs)
    fars = np.array(fars)
    
    # Find EER
    eer_threshold_idx = np.argmin(np.abs(frrs - fars))
    eer = (frrs[eer_threshold_idx] + fars[eer_threshold_idx]) / 2
    
    return {
        'frr_min': np.min(frrs),
        'frr_max': np.max(frrs),
        'frr_avg': np.mean(frrs),
        'far_min': np.min(fars),
        'far_max': np.max(fars),
        'far_avg': np.mean(fars),
        'eer': eer
    }

def create_results_table(train_results, test_results):
    """Create and save results table"""
    methods = ['method1', 'method2', 'method3', 'hybrid']
    rows = []
    
    for method in methods:
        # Calculate error rates for test set
        error_rates = calculate_error_rates(
            test_results[method]['genuine'],
            test_results[method]['impostor']
        )
        
        rows.append({
            'Method': method,
            'FRR Avg': f"{error_rates['frr_avg']:.3f}",
            'FRR Min': f"{error_rates['frr_min']:.3f}",
            'FRR Max': f"{error_rates['frr_max']:.3f}",
            'FAR Avg': f"{error_rates['far_avg']:.3f}",
            'FAR Min': f"{error_rates['far_min']:.3f}",
            'FAR Max': f"{error_rates['far_max']:.3f}",
            'EER': f"{error_rates['eer']:.3f}"
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv('results/error_rates.csv', index=False)
    print("\nResults saved to results/error_rates.csv")
    
    return df

def plot_roc_curves(results):
    """Plot ROC curves for all methods"""
    plt.figure(figsize=(10, 8))
    
    for method in results.keys():
        genuine_scores = results[method]['genuine']
        impostor_scores = results[method]['impostor']
        
        thresholds = np.linspace(0, 1, 100)
        tpr = []  # True Positive Rate
        fpr = []  # False Positive Rate
        
        for threshold in thresholds:
            tp = np.sum(np.array(genuine_scores) >= threshold)
            fn = np.sum(np.array(genuine_scores) < threshold)
            fp = np.sum(np.array(impostor_scores) >= threshold)
            tn = np.sum(np.array(impostor_scores) < threshold)
            
            tpr.append(tp / (tp + fn))
            fpr.append(fp / (fp + tn))
        
        plt.plot(fpr, tpr, label=method)
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Methods')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/roc_curves.png')
    print("ROC curves saved to visualizations/roc_curves.png")

def main():
    # Create required directories if they don't exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    print("Fingerprint Authentication System")
    print("--------------------------------")
    
    # Initialize processors
    processor = FingerprintProcessor()
    hybrid = HybridFingerprintMatcher()
    
    # Process training data
    train_results = process_dataset(processor, hybrid, 'train')
    
    # Process test data
    test_results = process_dataset(processor, hybrid, 'test')
    
    # Generate results table
    print("\nGenerating results table...")
    results_df = create_results_table(train_results, test_results)
    print("\nResults Table:")
    print(results_df)
    
    # Create visualizations
    print("\nGenerating ROC curves...")
    plot_roc_curves(test_results)
    
    print("\nProcessing complete! Check results/ and visualizations/ folders for output.")

if __name__ == "__main__":
    main()