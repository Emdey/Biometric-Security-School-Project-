"""
CYB 302 - Biometrics Security Lab
MIVA Open University
Complete Pipeline: Tasks 2-8
Dataset: ORL Face Dataset (AT&T) - 40 subjects, 10 images each
"""

import cv2
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import cosine
from cryptography.fernet import Fernet
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_PATH = "dataset"
OUTPUT_DIR   = "lab_outputs"
IMG_SIZE     = (100, 100)
os.makedirs(OUTPUT_DIR, exist_ok=True) #This creates the lab_outputs/ folder if it doesn't already exist. exist_ok=True means if the folder already exists, don't crash — just continue. Without this line, the code would crash the first time you run it because there's nowhere to save the plots.

# ============================================================
# TASK 2: PREPROCESSING  (your existing code, now as a function)
# ============================================================
def preprocess_image(img_path):
    original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        return None, None
    processed = cv2.resize(original, IMG_SIZE)
    processed = cv2.equalizeHist(processed)
    processed = cv2.GaussianBlur(processed, (3, 3), 0)
    original_resized = cv2.resize(original, IMG_SIZE)
    return original_resized, processed

def save_preprocessing_sample():
    """Save a before/after comparison image for the lab report."""
    for person in sorted(os.listdir(DATASET_PATH)):
        person_path = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(person_path):
            continue
        for img_name in sorted(os.listdir(person_path)):
            img_path = os.path.join(person_path, img_name)
            orig, proc = preprocess_image(img_path)
            if orig is None:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(orig, cmap='gray'); axes[0].set_title('Original'); axes[0].axis('off')
            axes[1].imshow(proc, cmap='gray'); axes[1].set_title('Preprocessed\n(Hist-EQ + Blur)'); axes[1].axis('off')
            plt.suptitle('Task 2: Preprocessing Comparison', fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'task2_preprocessing.png'), dpi=150)
            plt.close()
            print("[Task 2] Preprocessing sample saved.")
            return
        break

# ============================================================
# TASK 3: FEATURE EXTRACTION (PCA-based Eigenfaces)
# ============================================================
def load_all_images():
    """Load every image in the dataset; return flat vectors + labels."""
    images, labels = [], []
    persons = sorted([p for p in os.listdir(DATASET_PATH)
                      if os.path.isdir(os.path.join(DATASET_PATH, p))])
    for label_idx, person in enumerate(persons):
        person_path = os.path.join(DATASET_PATH, person)
        for img_name in sorted(os.listdir(person_path)):
            img_path = os.path.join(person_path, img_name)
            _, proc = preprocess_image(img_path)
            if proc is not None:
                images.append(proc.flatten().astype(np.float32))
                labels.append(label_idx)
    return np.array(images), np.array(labels), persons
   
def build_pca_templates(images, n_components=50):
    """Compute PCA and project all images into eigenface space."""
    mean_face = np.mean(images, axis=0)
    centered  = images - mean_face
    # Economy SVD (faster than full eigen-decomposition for this size)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    components = Vt[:n_components]          # principal components
    templates  = centered @ components.T    # projected feature vectors
    return templates, mean_face, components
 
def save_eigenfaces(mean_face, components):
    """Visualise the first 8 eigenfaces."""
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    # Mean face
    mf = mean_face.reshape(IMG_SIZE)
    mf = ((mf - mf.min()) / (mf.max() - mf.min()) * 255).astype(np.uint8)
    axes[0].imshow(mf, cmap='gray'); axes[0].set_title('Mean Face'); axes[0].axis('off')
    for i in range(1, 8):
        ef = components[i].reshape(IMG_SIZE)
        ef = ((ef - ef.min()) / (ef.max() - ef.min()) * 255).astype(np.uint8)
        axes[i].imshow(ef, cmap='gray')
        axes[i].set_title(f'Eigenface {i+1}')
        axes[i].axis('off')
    plt.suptitle('Task 3: Eigenfaces (PCA Feature Extraction)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'task3_eigenfaces.png'), dpi=150)
    plt.close()
    print("[Task 3] Eigenfaces saved.")

# ============================================================
# TASK 4: MATCHING  (Euclidean distance, 1:1 and 1:N)
# ============================================================
def split_enrolment_probe(templates, labels, enrol_per_subject=5):
    """First N images per subject → enrolment; rest → probe."""
    enrol_t, enrol_l, probe_t, probe_l = [], [], [], []
    unique = np.unique(labels)
    counts = {u: 0 for u in unique}
    for i, lbl in enumerate(labels):
        if counts[lbl] < enrol_per_subject:
            enrol_t.append(templates[i]); enrol_l.append(lbl)
            counts[lbl] += 1
        else:
            probe_t.append(templates[i]); probe_l.append(lbl)
    return (np.array(enrol_t), np.array(enrol_l),
            np.array(probe_t), np.array(probe_l))

def compute_all_scores(enrol_t, enrol_l, probe_t, probe_l):
    """
    Returns:
        genuine_scores  – distances where probe_label == enrol_label
        impostor_scores – distances where probe_label != enrol_label
        all pairs for ROC
    """
    genuine, impostor = [], []
    all_scores, all_labels_bin = [], []

    # Build mean enrolment template per subject
    unique = np.unique(enrol_l)
    gallery = {lbl: np.mean(enrol_t[enrol_l == lbl], axis=0) for lbl in unique}

    for i, (pt, pl) in enumerate(zip(probe_t, probe_l)):
        for lbl, et in gallery.items():
            dist = float(np.linalg.norm(pt - et))   # Euclidean distance
            is_genuine = int(pl == lbl)
            all_scores.append(dist); all_labels_bin.append(is_genuine)
            if is_genuine:
                genuine.append(dist)
            else:
                impostor.append(dist)

    return genuine, impostor, all_scores, all_labels_bin, gallery

def save_score_distributions(genuine, impostor):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(genuine,  bins=50, alpha=0.65, color='steelblue', label='Genuine pairs',  density=True)
    ax.hist(impostor, bins=50, alpha=0.65, color='tomato',    label='Impostor pairs', density=True)
    ax.set_xlabel('Euclidean Distance (lower = more similar)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Task 4: Genuine vs Impostor Score Distributions', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'task4_score_distributions.png'), dpi=150)
    plt.close()
    print("[Task 4] Score distribution saved.")

# ============================================================
# TASK 5: THRESHOLD SELECTION
# ============================================================
def threshold_analysis(genuine, impostor):
    """Sweep threshold; record FAR & FRR at each point."""
    all_scores = genuine + impostor
    thresholds = np.linspace(min(all_scores), max(all_scores), 300)
    far_list, frr_list = [], []

    genuine_arr  = np.array(genuine)
    impostor_arr = np.array(impostor)

    for t in thresholds:
        far = np.mean(impostor_arr <= t)   # impostor accepted (dist <= threshold)
        frr = np.mean(genuine_arr  >  t)   # genuine rejected  (dist >  threshold)
        far_list.append(far); frr_list.append(frr)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, far_list, color='tomato',    lw=2, label='FAR (False Acceptance Rate)')
    ax.plot(thresholds, frr_list, color='steelblue', lw=2, label='FRR (False Rejection Rate)')

    # Find EER
    far_arr = np.array(far_list); frr_arr = np.array(frr_list)
    eer_idx = np.argmin(np.abs(far_arr - frr_arr))
    eer_threshold = thresholds[eer_idx]
    eer_value     = (far_arr[eer_idx] + frr_arr[eer_idx]) / 2

    ax.axvline(eer_threshold, color='green', linestyle='--', lw=1.5,
               label=f'EER threshold = {eer_threshold:.1f}')
    ax.scatter([eer_threshold], [eer_value], color='green', zorder=5, s=80)
    ax.annotate(f'EER ≈ {eer_value*100:.1f}%',
                xy=(eer_threshold, eer_value),
                xytext=(eer_threshold + (thresholds[-1]-thresholds[0])*0.05, eer_value + 0.03),
                fontsize=10, color='green')

    ax.set_xlabel('Decision Threshold', fontsize=11)
    ax.set_ylabel('Error Rate', fontsize=11)
    ax.set_title('Task 5 & 6: FAR / FRR vs Threshold + EER', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'task5_FAR_FRR_threshold.png'), dpi=150)
    plt.close()

    print(f"[Task 5] EER Threshold = {eer_threshold:.2f}")
    print(f"[Task 6] EER = {eer_value*100:.2f}%  |  FAR@EER = {far_arr[eer_idx]*100:.2f}%  |  FRR@EER = {frr_arr[eer_idx]*100:.2f}%")
    return eer_threshold, eer_value, thresholds, far_list, frr_list

# ============================================================
# TASK 6: ROC CURVE
# ============================================================
def save_roc_curve(all_scores, all_labels_bin):
    # ROC expects higher score = more likely match; invert distance
    inv_scores = [-s for s in all_scores]
    fpr, tpr, _ = roc_curve(all_labels_bin, inv_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0,1],[0,1], 'k--', lw=1, label='Random guess')
    ax.set_xlabel('False Positive Rate (FAR)', fontsize=11)
    ax.set_ylabel('True Positive Rate (1 - FRR)', fontsize=11)
    ax.set_title('Task 6: ROC Curve', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'task6_roc_curve.png'), dpi=150)
    plt.close()
    print(f"[Task 6] ROC AUC = {roc_auc:.4f}")
    return roc_auc

# ============================================================
# TASK 7: MULTIMODAL FUSION
# ============================================================
def multimodal_fusion(templates, labels, n_comp_a=50, n_comp_b=30):
    """
    Simulate two modalities from the same dataset:
    Modality A: top 50 PCA components  (as already computed)
    Modality B: top 30 PCA components  (different subspace = second 'modality')
    Score-level fusion via weighted sum.
    """
    enrol_per = 5
    # Modality A
    tA, mA, cA = build_pca_templates(templates, n_comp_a)
    enA_t, enA_l, prA_t, prA_l = split_enrolment_probe(tA, labels, enrol_per)
    gA, iA, _, _, galA = compute_all_scores(enA_t, enA_l, prA_t, prA_l)

    # Modality B (fewer components, simulates a second weaker sensor)
    tB, mB, cB = build_pca_templates(templates, n_comp_b)
    enB_t, enB_l, prB_t, prB_l = split_enrolment_probe(tB, labels, enrol_per)
    gB, iB, _, _, galB = compute_all_scores(enB_t, enB_l, prB_t, prB_l)

    # Normalise scores to [0,1] then fuse
    def normalise(arr):
        arr = np.array(arr)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-9)

    gA_n = normalise(gA); gB_n = normalise(gB)
    iA_n = normalise(iA); iB_n = normalise(iB)
    min_len_g = min(len(gA_n), len(gB_n))
    min_len_i = min(len(iA_n), len(iB_n))
    fused_genuine  = 0.6 * gA_n[:min_len_g] + 0.4 * gB_n[:min_len_g]
    fused_impostor = 0.6 * iA_n[:min_len_i] + 0.4 * iB_n[:min_len_i]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, g, imp, title in zip(
        axes,
        [gA, gB, fused_genuine.tolist()],
        [iA, iB, fused_impostor.tolist()],
        ['Modality A (50 PCs)', 'Modality B (30 PCs)', 'Fused (Weighted Sum)']
    ):
        ax.hist(g,   bins=40, alpha=0.65, color='steelblue', label='Genuine',  density=True)
        ax.hist(imp, bins=40, alpha=0.65, color='tomato',    label='Impostor', density=True)
        ax.set_title(title); ax.legend(fontsize=9)
        ax.set_xlabel('Distance / Fused Score')
    plt.suptitle('Task 7: Multimodal Fusion Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'task7_multimodal_fusion.png'), dpi=150)
    plt.close()
    print("[Task 7] Multimodal fusion plot saved.")

# ============================================================
# TASK 8: SECURITY – TEMPLATE ENCRYPTION
# ============================================================
def encrypt_templates(gallery):
    """Encrypt biometric templates with Fernet symmetric encryption."""
    key   = Fernet.generate_key()
    fernet = Fernet(key)

    encrypted = {}
    for lbl, template in gallery.items():
        raw_bytes  = template.astype(np.float32).tobytes()
        enc_bytes  = fernet.encrypt(raw_bytes)
        encrypted[int(lbl)] = enc_bytes.decode()

    # Save key (in practice: store in a Hardware Security Module, never with data)
    key_path = os.path.join(OUTPUT_DIR, 'task8_encryption_key.key')
    with open(key_path, 'wb') as f:
        f.write(key)

    # Save encrypted templates
    enc_path = os.path.join(OUTPUT_DIR, 'task8_encrypted_templates.json')
    with open(enc_path, 'w') as f:
        json.dump(encrypted, f, indent=2)

    print(f"[Task 8] {len(encrypted)} templates encrypted and saved.")
    print(f"         Key stored at: {key_path}")
    print(f"         Templates at:  {enc_path}")

    # Demonstrate decryption round-trip
    first_lbl = list(encrypted.keys())[0]
    dec_bytes  = fernet.decrypt(encrypted[first_lbl].encode())
    recovered  = np.frombuffer(dec_bytes, dtype=np.float32)
    original   = gallery[first_lbl]
    match = np.allclose(recovered, original)
    print(f"[Task 8] Decryption round-trip check: {'PASSED' if match else 'FAILED'}")

# ============================================================
# SUMMARY REPORT  (printed to console + saved as txt)
# ============================================================
def save_summary(eer_value, roc_auc, n_subjects, n_images):
    lines = [
        "=" * 60,
        "  CYB 302 BIOMETRICS LAB – RESULTS SUMMARY",
        "=" * 60,
        f"  Dataset          : ORL Face (AT&T)",
        f"  Subjects         : {n_subjects}",
        f"  Total images     : {n_images}",
        f"  Image size       : {IMG_SIZE[0]}x{IMG_SIZE[1]} px",
        f"  Feature method   : PCA (Eigenfaces, 50 components)",
        f"  Matching metric  : Euclidean distance",
        "",
        "  PERFORMANCE METRICS",
        f"  Equal Error Rate (EER)  : {eer_value*100:.2f}%",
        f"  ROC AUC                 : {roc_auc:.4f}",
        "",
        "  OUTPUT FILES",
        "  task2_preprocessing.png",
        "  task3_eigenfaces.png",
        "  task4_score_distributions.png",
        "  task5_FAR_FRR_threshold.png",
        "  task6_roc_curve.png",
        "  task7_multimodal_fusion.png",
        "  task8_encrypted_templates.json",
        "  task8_encryption_key.key",
        "=" * 60,
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(OUTPUT_DIR, "summary_report.txt"), "w") as f:
        f.write(report)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  CYB 302 - Biometrics Lab | Full Pipeline")
    print("=" * 55)

    # Task 2 – save preprocessing sample image
    save_preprocessing_sample()

    # Load all images
    print("\n[Loading] Reading dataset...")
    raw_images, labels, persons = load_all_images()
    print(f"[Loading] {len(raw_images)} images loaded across {len(persons)} subjects.")

    # Task 3 – PCA feature extraction
    print("\n[Task 3] Extracting PCA features...")
    templates, mean_face, components = build_pca_templates(raw_images, n_components=50)
    save_eigenfaces(mean_face, components)
    print(f"[Task 3] Template shape per image: {templates[0].shape}  (50-dim vector)")

    # Task 4 – Matching & score computation
    print("\n[Task 4] Computing genuine/impostor scores...")
    enrol_t, enrol_l, probe_t, probe_l = split_enrolment_probe(templates, labels)
    genuine, impostor, all_scores, all_labels_bin, gallery = \
        compute_all_scores(enrol_t, enrol_l, probe_t, probe_l)
    save_score_distributions(genuine, impostor)
    print(f"[Task 4] Genuine pairs: {len(genuine)} | Impostor pairs: {len(impostor)}")

    # Tasks 5 & 6 – Threshold + EER
    print("\n[Task 5/6] Threshold sweep and EER calculation...")
    eer_threshold, eer_value, thresholds, far_list, frr_list = \
        threshold_analysis(genuine, impostor)

    # Task 6 – ROC curve
    roc_auc = save_roc_curve(all_scores, all_labels_bin)

    # Task 7 – Multimodal fusion
    print("\n[Task 7] Multimodal fusion...")
    multimodal_fusion(raw_images, labels)

    # Task 8 – Encrypt templates
    print("\n[Task 8] Encrypting biometric templates...")
    encrypt_templates(gallery)

    # Summary
    save_summary(eer_value, roc_auc, len(persons), len(raw_images))

    print(f"\nDONE! All outputs saved to: ./{OUTPUT_DIR}/")