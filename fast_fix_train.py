import os
import re
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

# =====================================================================
# 1. LOAD CACHED ARTIFACTS
# =====================================================================
dataset_path = os.getcwd()
captions_file = os.path.join(dataset_path, "captions.txt")
features_file = os.path.join(dataset_path, "features.pkl")
model_file = os.path.join(dataset_path, "caption_model.keras")
tokenizer_file = os.path.join(dataset_path, "tokenizer.pkl")

print("Loading cached tokenizer, features, and model...")

with open(tokenizer_file, "rb") as f:
    tokenizer = pickle.load(f)

with open(features_file, "rb") as f:
    features = pickle.load(f)

model = load_model(model_file)

# Match target shape exactly to the loaded model's final Dense layer
vocab_size = model.output_shape[-1]
max_length = 37

print(f"Target dimension set to match model output: {vocab_size}")

# =====================================================================
# 2. LOAD & FORMAT TRAINING CAPTIONS
# =====================================================================
def clean_text(text):
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()

train_samples = []
with open(captions_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

start_idx = 1 if "image" in lines[0].lower() else 0
for line in lines[start_idx:]:
    parts = line.strip().split(",")
    if len(parts) < 2:
        continue
    img_id = parts[0].strip()
    raw_cap = ",".join(parts[1:]).strip()
    if img_id in features:
        train_samples.append((img_id, f"start {clean_text(raw_cap)} end"))

print(f"Prepared {len(train_samples)} training pairs.")

# =====================================================================
# 3. FAST BATCH GENERATOR (WITH SHAPE-COMPLIANT TARGETS)
# =====================================================================
def fast_generator(samples, feat_dict, tok, max_len, v_size, batch_size=256):
    num_samples = len(samples)
    while True:
        np.random.shuffle(samples)
        for i in range(0, num_samples, batch_size):
            batch = samples[i : i + batch_size]
            X1, X2, y = [], [], []
            for img_id, caption in batch:
                feat = feat_dict[img_id]
                seq = tok.texts_to_sequences([caption])[0]
                for j in range(1, len(seq)):
                    in_seq, out_seq = seq[:j], seq[j]
                    
                    # Ensure the token ID falls strictly within the model's vocabulary dimension
                    if out_seq >= v_size:
                        continue
                        
                    in_seq = pad_sequences([in_seq], maxlen=max_len, padding="post")[0]
                    out_seq = to_categorical([out_seq], num_classes=v_size)[0]
                    
                    X1.append(feat)
                    X2.append(in_seq)
                    y.append(out_seq)
                    
            if len(X1) > 0:
                yield {"image_input": np.array(X1), "text_input": np.array(X2)}, np.array(y)

# =====================================================================
# 4. RAPID TRAINING EXECUTION
# =====================================================================
batch_size = 256
steps = min(len(train_samples) // batch_size, 150)

# Compile using Adam optimizer with gradient clipping
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.0005, clipnorm=1.0))

lr_reducer = ReduceLROnPlateau(monitor="loss", factor=0.5, patience=1, verbose=1)

print("\nStarting rapid alignment training for 5 epochs...")
gen = fast_generator(train_samples, features, tokenizer, max_length, vocab_size, batch_size)

model.fit(gen, steps_per_epoch=steps, epochs=5, callbacks=[lr_reducer], verbose=1)

# Save the updated weights directly to caption_model.keras
model.save(model_file)
print("\nSuccess! Saved improved weights to 'caption_model.keras'.")