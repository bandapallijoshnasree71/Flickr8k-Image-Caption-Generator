import os
import re
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, add, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

warnings.filterwarnings('ignore')

# =====================================================================
# STEP 1: PATH CONFIGURATION
# =====================================================================
dataset_path = os.getcwd()
images_directory = os.path.join(dataset_path, "Images")
captions_path = os.path.join(dataset_path, "captions.txt")
features_path = os.path.join(dataset_path, "features.pkl")
model_path = os.path.join(dataset_path, "caption_model.keras")
tokenizer_path = os.path.join(dataset_path, "tokenizer.pkl")

# Fallback for cloud/Kaggle environments
if not os.path.exists(images_directory) and os.path.exists("/kaggle/input/flickr8k/Images"):
    images_directory = "/kaggle/input/flickr8k/Images/"
    captions_path = "/kaggle/input/flickr8k/captions.txt"

print(f"Working Directory: {dataset_path}")
print(f"Images Directory:  {images_directory}")
print(f"Captions Path:     {captions_path}")

# =====================================================================
# STEPS 2, 3 & 4: LOAD, CLEAN, AND FORMAT CAPTIONS
# =====================================================================
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_captions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    start_idx = 1 if 'image' in lines[0].lower() else 0
    captions_IDs = []
    cleaned_captions_list = []
    descriptions_dict = {}

    for line in lines[start_idx:]:
        tokens = line.strip().split(',')
        if len(tokens) < 2:
            continue
        image_id = tokens[0].strip()
        raw_caption = ",".join(tokens[1:]).strip().lower()
        cleaned = clean_text(raw_caption)
        
        formatted_entry = f"{image_id}\tstart {cleaned} end\n"
        captions_IDs.append(formatted_entry)
        cleaned_captions_list.append(cleaned)
        
        if image_id not in descriptions_dict:
            descriptions_dict[image_id] = []
        descriptions_dict[image_id].append(f"start {cleaned} end")
        
    return captions_IDs, cleaned_captions_list, descriptions_dict

captions_IDs, cleaned_captions, descriptions_dict = load_and_preprocess_captions(captions_path)
print(f"Loaded {len(captions_IDs)} captions across {len(descriptions_dict)} images.")

# =====================================================================
# STEP 6: EXPLORING CAPTION LENGTH DISTRIBUTION (PURE MATPLOTLIB)
# =====================================================================
def plot_captions_length(data, save_path="caption_length_histogram.png"):
    lengths = [len(x.split(' ')) for x in data]
    plt.figure(figsize=(10, 5), dpi=150)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.hist(lengths, bins=range(min(lengths), max(lengths) + 1), color='#2980B9', edgecolor='black')
    plt.title('Captions Length Histogram', fontsize=14, fontweight='bold')
    plt.xlabel('Length (Words)', fontweight='bold')
    plt.ylabel('Frequency', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Caption length distribution saved to '{save_path}'.")
    plt.close()

plot_captions_length(cleaned_captions)

# =====================================================================
# STEP 7: TOKENIZATION
# =====================================================================
if os.path.exists(tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    print("Loaded cached tokenizer from tokenizer.pkl.")
else:
    all_wrapped_captions = [cap.split('\t')[1].strip() for cap in captions_IDs]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_wrapped_captions)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print("Fitted and saved new tokenizer to tokenizer.pkl.")

vocab_size = len(tokenizer.word_index) + 1
max_caption_length = max(len(caption.split('\t')[1].strip().split()) for caption in captions_IDs)
print(f"Vocabulary Size: {vocab_size} | Max Caption Length: {max_caption_length}")

# =====================================================================
# STEP 8: DATASET SPLIT (TRAIN / VALIDATION / TEST)
# =====================================================================
all_image_ids = [img for img in os.listdir(images_directory) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
train_image_ids, val_image_ids = train_test_split(all_image_ids, test_size=0.15, random_state=42)
val_image_ids, test_image_ids = train_test_split(val_image_ids, test_size=0.10, random_state=42)

train_captions, val_captions, test_captions = [], [], []
for caption in captions_IDs:
    image_id, _ = caption.split('\t')
    if image_id in train_image_ids:
        train_captions.append(caption)
    elif image_id in val_image_ids:
        val_captions.append(caption)
    elif image_id in test_image_ids:
        test_captions.append(caption)

print(f"Split: {len(train_image_ids)} Train | {len(val_image_ids)} Val | {len(test_image_ids)} Test images.")

# =====================================================================
# STEPS 9 & 10: INCEPTIONV3 FEATURE EXTRACTION & CACHING
# =====================================================================
def build_feature_extractor():
    base = InceptionV3(weights='imagenet', input_shape=(299, 299, 3))
    feature_model = Model(inputs=base.inputs, outputs=base.layers[-2].output)
    return feature_model

def extract_image_features(model, img_path):
    img = load_img(img_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img, verbose=0)
    return features.flatten()

if os.path.exists(features_path):
    print("Loading cached image features from features.pkl...")
    with open(features_path, 'rb') as f:
        image_features_dict = pickle.load(f)
else:
    print("Extracting features using InceptionV3 (one-time execution)...")
    extractor = build_feature_extractor()
    image_features_dict = {}
    for idx, img_id in enumerate(all_image_ids):
        p = os.path.join(images_directory, img_id)
        try:
            image_features_dict[img_id] = extract_image_features(extractor, p)
        except Exception:
            continue
        if (idx + 1) % 1000 == 0:
            print(f"Extracted {idx + 1}/{len(all_image_ids)} images...")
    with open(features_path, 'wb') as f:
        pickle.dump(image_features_dict, f)
    print("Saved features.pkl successfully.")

train_image_features = {k: image_features_dict[k] for k in train_image_ids if k in image_features_dict}
val_image_features   = {k: image_features_dict[k] for k in val_image_ids if k in image_features_dict}
test_image_features  = {k: image_features_dict[k] for k in test_image_ids if k in image_features_dict}

# =====================================================================
# STEP 11: BATCH DATA GENERATOR
# =====================================================================
def data_generator(captions, image_features, tok, max_len, v_size, batch_size):
    num_samples = len(captions)
    while True:
        np.random.shuffle(captions)
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            X_images, X_captions, y = [], [], []
            for caption in captions[start_idx:end_idx]:
                image_id, caption_text = caption.split('\t')
                caption_text = caption_text.rstrip('\n')
                if image_id not in image_features:
                    continue
                seq = tok.texts_to_sequences([caption_text])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_len, padding='post')[0]
                    out_seq = to_categorical([out_seq], num_classes=v_size)[0]
                    X_images.append(image_features[image_id])
                    X_captions.append(in_seq)
                    y.append(out_seq)

            if len(X_images) > 0:
                yield [np.array(X_images), np.array(X_captions)], np.array(y)

cnn_output_dim = 2048
batch_size_train = 128
batch_size_val = 64

train_data_generator = data_generator(train_captions, train_image_features, tokenizer, max_caption_length, vocab_size, batch_size_train)
val_data_generator = data_generator(val_captions, val_image_features, tokenizer, max_caption_length, vocab_size, batch_size_val)

# =====================================================================
# STEP 12: MODEL ARCHITECTURE
# =====================================================================
def build_model(v_size, max_len, cnn_dim):
    # Image pipeline with Batch Normalization
    input_image = Input(shape=(cnn_dim,), name='Features_Input')
    fe1 = BatchNormalization()(input_image)
    fe2 = Dense(256, activation='relu')(fe1)
    fe3 = BatchNormalization()(fe2)

    # Sequence pipeline with Embedding & LSTM
    input_caption = Input(shape=(max_len,), name='Sequence_Input')
    se1 = Embedding(v_size, 256, mask_zero=True)(input_caption)
    se2 = LSTM(256)(se1)

    # Combined Decoder
    decoder1 = add([fe3, se2])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(v_size, activation='softmax', name='Output_Layer')(decoder2)

    m = Model(inputs=[input_image, input_caption], outputs=outputs, name='Image_Captioning')
    return m

if os.path.exists(model_path):
    print("Loading pre-trained caption model from caption_model.keras...")
    caption_model = load_model(model_path)
else:
    print("Building new CNN-LSTM architecture...")
    caption_model = build_model(vocab_size, max_caption_length, cnn_output_dim)
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    caption_model.compile(loss='categorical_crossentropy', optimizer=optimizer)

caption_model.summary()

# =====================================================================
# STEPS 13 & 14: TRAINING WITH EARLY STOPPING & SCHEDULER
# =====================================================================
if not os.path.exists(model_path):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    def lr_scheduler(epoch, lr):
        if epoch < 3:
            return lr
        return lr * tf.math.exp(-0.2)

    lr_schedule = LearningRateScheduler(lr_scheduler)

    steps_train = len(train_captions) // batch_size_train
    steps_val = len(val_captions) // batch_size_val

    print(f"Starting model training for 15 epochs ({steps_train} train steps/epoch)...")
    history = caption_model.fit(
        train_data_generator,
        steps_per_epoch=steps_train,
        validation_data=val_data_generator,
        validation_steps=steps_val,
        epochs=15,
        callbacks=[early_stopping, lr_schedule]
    )
    caption_model.save(model_path)
    print("Saved trained model to caption_model.keras.")

    # Plot Loss Curves using pure Matplotlib
    plt.figure(figsize=(10, 5), dpi=150)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.plot(history.history['loss'], color='#E74C3C', marker='o', label='Train Loss')
    plt.plot(history.history['val_loss'], color='#641E16', marker='h', label='Validation Loss')
    plt.title('Train vs Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.close()

# =====================================================================
# STEP 15: GREEDY CAPTION GENERATION
# =====================================================================
def greedy_generator(image_features, max_len=max_caption_length):
    in_text = 'start'
    for _ in range(max_len):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_caption_length, padding='post')
        prediction = caption_model.predict([image_features.reshape(1, cnn_output_dim), sequence], verbose=0)[0]
        idx = int(np.argmax(prediction))
        word = tokenizer.index_word.get(idx, None)
        
        if word is None or word == 'end':
            break
        in_text += ' ' + word

    return in_text.replace('start', '').strip()

# =====================================================================
# STEP 16: BEAM SEARCH GENERATION (WITH REPETITION PENALTY)
# =====================================================================
def beam_search_generator(image_features, K_beams=3, max_len=max_caption_length):
    start = [tokenizer.word_index['start']]
    start_word = [[start, 0.0]]
    
    for _ in range(max_len):
        temp = []
        for s in start_word:
            if s[0][-1] == tokenizer.word_index.get('end', -1):
                temp.append(s)
                continue
                
            sequence = pad_sequences([s[0]], maxlen=max_caption_length, padding='post')
            preds = caption_model.predict([image_features.reshape(1, cnn_output_dim), sequence], verbose=0)[0]
            
            # Penalize immediate consecutive repetitions
            if len(s[0]) > 1:
                preds[s[0][-1]] *= 0.1
                
            log_preds = np.log(np.maximum(preds, 1e-10))
            word_preds = np.argsort(log_preds)[-K_beams:]
            
            for w in word_preds:
                next_cap = s[0] + [w]
                prob = s[1] + log_preds[w]
                temp.append([next_cap, prob])
                
        start_word = sorted(temp, key=lambda l: l[1], reverse=True)[:K_beams]
        if all(s[0][-1] == tokenizer.word_index.get('end', -1) for s in start_word):
            break

    best_sequence = start_word[0][0]
    words = [tokenizer.index_word.get(i, '') for i in best_sequence if i not in (tokenizer.word_index['start'], tokenizer.word_index.get('end', -1))]
    return " ".join([w for w in words if w])

# =====================================================================
# STEP 17: BLEU SCORE EVALUATION (CONSISTENT LABELS)
# =====================================================================
def evaluate_bleu(actual_captions, greedy_pred, beam_pred):
    smooth = SmoothingFunction().method1
    refs = [cap.replace('start ', '').replace(' end', '').split() for cap in actual_captions]
    g_tokens = greedy_pred.split()
    b_tokens = beam_pred.split()

    bleu1_greedy = sentence_bleu(refs, g_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    bleu2_greedy = sentence_bleu(refs, g_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    bleu1_beam   = sentence_bleu(refs, b_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    bleu2_beam   = sentence_bleu(refs, b_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)

    return (
        f"Beam Search: {beam_pred}",
        f"BLEU-1 Beam Search: {bleu1_beam:.5f}",
        f"BLEU-2 Beam Search: {bleu2_beam:.5f}",
        f"Greedy: {greedy_pred}",
        f"BLEU-1 Greedy: {bleu1_greedy:.5f}",
        f"BLEU-2 Greedy: {bleu2_greedy:.5f}"
    )

# =====================================================================
# STEP 18: TARGET 1 - DATASET EXPLORATION VIEW
# =====================================================================
def display_dataset_exploration(mapping, img_folder, num_samples=5):
    """
    Renders sample images on the left and their 5 ground-truth captions on the right.
    Anchors each caption to an independent Y-coordinate to avoid overlap.
    """
    from matplotlib.gridspec import GridSpec

    sample_images = [img for img in mapping.keys() if os.path.exists(os.path.join(img_folder, img))][:num_samples]

    fig = plt.figure(figsize=(11, num_samples * 2.7), dpi=150)
    gs = GridSpec(
        nrows=num_samples,
        ncols=2,
        figure=fig,
        width_ratios=[1.2, 3.8],
        wspace=0.05,
        hspace=0.25
    )

    for i, img_name in enumerate(sample_images):
        ax_img = fig.add_subplot(gs[i, 0])
        ax_txt = fig.add_subplot(gs[i, 1])

        # Image display
        img = Image.open(os.path.join(img_folder, img_name))
        ax_img.imshow(img)
        ax_img.axis("off")

        # Captions display
        ax_txt.axis("off")
        ax_txt.set_xlim(0, 1)
        ax_txt.set_ylim(0, 1)

        captions = mapping[img_name]
        y_coords = [0.88, 0.69, 0.50, 0.31, 0.12]

        for cap_idx, text in enumerate(captions[:5]):
            ax_txt.text(
                0.03, y_coords[cap_idx], text,
                fontsize=9.5,
                verticalalignment="center",
                horizontalalignment="left",
                color="#111111",
                fontfamily="sans-serif"
            )

    exploration_file = "dataset_exploration_clean.png"
    plt.savefig(exploration_file, bbox_inches="tight", dpi=200, pad_inches=0.1)
    print(f"Saved clean exploration view to '{exploration_file}'.")

    try:
        manager = plt.get_current_fig_manager()
        if hasattr(manager.window, "state"):
            manager.window.state("zoomed")
        elif hasattr(manager.window, "showMaximized"):
            manager.window.showMaximized()
    except Exception:
        pass

    plt.show(block=True)


# =====================================================================
# STEP 19: TARGET 2 - PREDICTION & BLEU EVALUATION REPORT
# =====================================================================
def visualize_results(test_keys, mapping, feat_dict, img_dir, num_of_images=7):
    """
    Renders 7 test images on the left and the 6 metric lines on the right,
    balanced directly alongside each image without text overlap.
    """
    from matplotlib.gridspec import GridSpec

    selected_images = [img for img in test_keys if os.path.exists(os.path.join(img_dir, img))][:num_of_images]

    fig = plt.figure(figsize=(10, num_of_images * 2.2), dpi=150)
    gs = GridSpec(
        nrows=num_of_images,
        ncols=2,
        figure=fig,
        width_ratios=[1.0, 2.8],
        wspace=0.04,
        hspace=0.35
    )

    for row, img_name in enumerate(selected_images):
        img_path = os.path.join(img_dir, img_name)
        feat = feat_dict[img_name]
        actual_caps = mapping.get(img_name, [])

        greedy_cap = greedy_generator(feat)
        beam_cap = beam_search_generator(feat, K_beams=3)
        metrics = evaluate_bleu(actual_caps, greedy_cap, beam_cap)

        # Left: Image
        ax_img = fig.add_subplot(gs[row, 0])
        ax_img.imshow(Image.open(img_path))
        ax_img.axis("off")

        # Right: Formatted metrics
        ax_txt = fig.add_subplot(gs[row, 1])
        ax_txt.axis("off")
        ax_txt.set_xlim(0, 1)
        ax_txt.set_ylim(0, 1)

        y_positions = [0.88, 0.72, 0.56, 0.40, 0.24, 0.08]
        for text_str, y_pos in zip(metrics, y_positions):
            ax_txt.text(
                0.02, y_pos, text_str,
                fontsize=8.5,
                verticalalignment="center",
                horizontalalignment="left",
                color="#111111",
                fontfamily="sans-serif"
            )

    report_output_file = "model_evaluation_report_clean.png"
    plt.savefig(report_output_file, bbox_inches="tight", dpi=200, pad_inches=0.08)
    print(f"Saved clean evaluation report to '{report_output_file}'.")

    try:
        manager = plt.get_current_fig_manager()
        if hasattr(manager.window, "state"):
            manager.window.state("zoomed")
        elif hasattr(manager.window, "showMaximized"):
            manager.window.showMaximized()
    except Exception:
        pass

    plt.show(block=True)


# =====================================================================
# MAIN EXECUTION: DISPLAY BOTH TARGET 1 AND TARGET 2
# =====================================================================
if __name__ == "__main__":
    print("\n--- Displaying Target 1: Dataset Exploration ---")
    display_dataset_exploration(descriptions_dict, images_directory, num_samples=5)

    print("\n--- Displaying Target 2: Model Prediction & Evaluation Report ---")
    visualize_results(test_image_ids, descriptions_dict, test_image_features, images_directory, num_of_images=7)