import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================================
# STEP 1: LOAD MODEL, TOKENIZER, AND CONFIGURE PATHS
# =====================================================================
dataset_path = os.getcwd()
images_dir = os.path.join(dataset_path, "Images")
model_path = os.path.join(dataset_path, "caption_model.keras")
tokenizer_path = os.path.join(dataset_path, "tokenizer.pkl")

print("=" * 60)
print("Loading trained captioning model and tokenizer...")
print("=" * 60)

if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
    raise FileNotFoundError("Missing caption_model.keras or tokenizer.pkl in the current directory.")

model = load_model(model_path)

with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}
max_caption_length = 37

# =====================================================================
# STEP 2: INCEPTIONV3 FEATURE EXTRACTOR
# =====================================================================
print("Initializing InceptionV3 feature extractor...")
base_model = InceptionV3(weights="imagenet")
feature_extractor = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

def extract_features(img_path):
    """Loads an image, converts to RGB, resizes to (299, 299), and extracts 2048-d vector."""
    img = Image.open(img_path).convert("RGB").resize((299, 299))
    arr = np.expand_dims(np.array(img), axis=0)
    arr = preprocess_input(arr)
    return feature_extractor.predict(arr, verbose=0).reshape((2048,))

# =====================================================================
# STEP 3: ENHANCED DECODERS WITH STRICT N-GRAM REPETITION SUPPRESSION
# =====================================================================
def apply_repetition_penalty(preds, tokens, tok_index, no_repeat_ngram_size=2):
    """
    Sets logits to 0 for words that would repeat immediate single words
    or create recurring n-gram phrases (e.g. 'a grass a grass').
    """
    # 1. Block immediate consecutive word repeat
    if len(tokens) > 1 and tokens[-1] in tok_index:
        preds[tok_index[tokens[-1]]] = 0.0

    # 2. Block 2-gram repetitions
    if len(tokens) >= no_repeat_ngram_size:
        prefix = tuple(tokens[-(no_repeat_ngram_size - 1):])
        for i in range(len(tokens) - no_repeat_ngram_size + 1):
            cur_prefix = tuple(tokens[i : i + no_repeat_ngram_size - 1])
            if cur_prefix == prefix:
                banned_word = tokens[i + no_repeat_ngram_size - 1]
                if banned_word in tok_index:
                    preds[tok_index[banned_word]] = 0.0

    # 3. Penalize general frequency of all previously generated words
    for word in set(tokens):
        if word in tok_index:
            count = tokens.count(word)
            preds[tok_index[word]] *= (0.5 ** count)

    return preds

def generate_greedy(feat, max_len=14):
    """Generates captions using argmax with strict anti-repetition guards."""
    tokens = ["start"]
    for _ in range(max_len):
        seq = tokenizer.texts_to_sequences([" ".join(tokens)])[0]
        seq = pad_sequences([seq], maxlen=max_caption_length, padding="post")
        preds = model.predict({"image_input": feat.reshape(1, 2048), "text_input": seq}, verbose=0)[0]

        preds = apply_repetition_penalty(preds, tokens, tokenizer.word_index, no_repeat_ngram_size=2)

        # Avoid divide-by-zero if all probabilities penalized
        prob_sum = np.sum(preds)
        if prob_sum == 0:
            break
        preds = preds / prob_sum

        next_idx = int(np.argmax(preds))
        word = index_to_word.get(next_idx, None)

        if word is None or word == "end":
            break
        tokens.append(word)

    clean_words = [w for w in tokens if w not in ("start", "end")]
    return " ".join(clean_words)

def generate_beam_search(feat, beam_width=3, max_len=14):
    """Maintains top-k hypothesis paths with normalized sequence length scoring."""
    sequences = [(["start"], 0.0)]

    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] == "end":
                all_candidates.append((seq, score))
                continue

            padded = pad_sequences(
                [tokenizer.texts_to_sequences([" ".join(seq)])[0]],
                maxlen=max_caption_length,
                padding="post"
            )
            preds = model.predict({"image_input": feat.reshape(1, 2048), "text_input": padded}, verbose=0)[0]
            preds = apply_repetition_penalty(preds, seq, tokenizer.word_index, no_repeat_ngram_size=2)

            log_preds = np.log(np.maximum(preds, 1e-10))
            top_k_indices = np.argsort(log_preds)[-beam_width:]

            for idx in top_k_indices:
                word = index_to_word.get(idx, "")
                candidate_seq = seq + [word]
                candidate_score = score + log_preds[idx]
                all_candidates.append((candidate_seq, candidate_score))

        # Length-normalized ranking to prevent run-on sentences
        ordered = sorted(
            all_candidates,
            key=lambda item: item[1] / ((len(item[0]) + 5) / 6.0) ** 0.7,
            reverse=True
        )
        sequences = ordered[:beam_width]

        if all(seq[-1] == "end" for seq, _ in sequences):
            break

    best_seq = sequences[0][0]
    clean_words = [w for w in best_seq if w not in ("start", "end")]
    return " ".join(clean_words)

# =====================================================================
# STEP 4: INTERACTIVE PREDICTION & DISPLAY FUNCTION
# =====================================================================
def run_caption_inference(image_path):
    if not os.path.exists(image_path):
        print(f"\n[Error]: File does not exist at '{image_path}'. Please check the path.")
        return

    print(f"\nAnalyzing image: {image_path} ...")
    feat = extract_features(image_path)

    greedy_caption = generate_greedy(feat)
    beam_caption = generate_beam_search(feat, beam_width=3)

    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"Greedy Caption     : {greedy_caption}")
    print(f"Beam Search Caption: {beam_caption}")
    print("=" * 60)

    # Display image with clean formatting
    plt.figure(figsize=(7, 7), dpi=120)
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis("off")
    title_text = f"Beam Search: {beam_caption}\nGreedy: {greedy_caption}"
    plt.title(title_text, fontsize=10.5, pad=12, fontweight="bold")
    plt.tight_layout()
    plt.show(block=True)

# =====================================================================
# STEP 5: EXECUTION ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    print("\nEnter the path to an image (e.g., Images/1000268201_693b08cb0e.jpg).")
    user_input = input("Press ENTER to test a random image from Images folder: ").strip().strip('"').strip("'")

    if user_input == "":
        if os.path.exists(images_dir):
            sample_images = [img for img in os.listdir(images_dir) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if sample_images:
                chosen = np.random.choice(sample_images)
                target_path = os.path.join(images_dir, chosen)
                run_caption_inference(target_path)
            else:
                print("No images found in Images folder.")
        else:
            print("Images directory not found.")
    else:
        run_caption_inference(user_input)