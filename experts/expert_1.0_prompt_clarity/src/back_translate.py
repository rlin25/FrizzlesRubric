import os
import glob
import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer

# ========== Translation Utilities ==========

def load_translation_models(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate(texts, tokenizer, model):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated = model.generate(**inputs)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

def back_translate_batch(texts, en_fr_tokenizer, en_fr_model, fr_en_tokenizer, fr_en_model):
    french = translate(texts, en_fr_tokenizer, en_fr_model)
    english = translate(french, fr_en_tokenizer, fr_en_model)
    return english

# ========== File Processing ==========

def process_directory(input_dir, clarity_label, output_dir,
                      en_fr_tokenizer, en_fr_model, fr_en_tokenizer, fr_en_model):
    
    os.makedirs(output_dir, exist_ok=True)
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        if "text" not in df.columns or "label" not in df.columns:
            print(f"⚠️ Skipping {file_path} — 'text' or 'label' column missing.")
            continue
        
        print(f"🔁 Back-translating {file_path} ...")
        prompts = df["text"].tolist()
        labels = df["label"].tolist()
        augmented_prompts = []

        # Process in batches to avoid memory issues
        batch_size = 16
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            augmented = back_translate_batch(batch, en_fr_tokenizer, en_fr_model, fr_en_tokenizer, fr_en_model)
            augmented_prompts.extend(augmented)

        # Save new CSV with the new name format
        output_df = pd.DataFrame({
            "text": augmented_prompts,
            "label": labels[:len(augmented_prompts)]  # Ensure the length matches
        })

        # Modify filename to the new format: clarity_<original_name>_back_translated.csv
        base_name = os.path.basename(file_path)
        new_name = base_name.replace(".csv", "_back_translated.csv")
        output_path = os.path.join(output_dir, new_name)
        
        output_df.to_csv(output_path, index=False)
        print(f"✅ Saved back-translated data to {output_path}")

# ========== Main Execution ==========

if __name__ == "__main__":
    # Language models
    print("📦 Loading translation models...")
    en_fr_tokenizer, en_fr_model = load_translation_models("en", "fr")
    fr_en_tokenizer, fr_en_model = load_translation_models("fr", "en")

    # Directories
    HIGH_DIR = "data/original_high_clarity"
    LOW_DIR = "data/original_low_clarity"
    OUTPUT_DIR = "data/augmented_backtranslated"

    # Process both datasets
    process_directory(HIGH_DIR, clarity_label=1, output_dir=OUTPUT_DIR,
                      en_fr_tokenizer=en_fr_tokenizer, en_fr_model=en_fr_model,
                      fr_en_tokenizer=fr_en_tokenizer, fr_en_model=fr_en_model)

    process_directory(LOW_DIR, clarity_label=0, output_dir=OUTPUT_DIR,
                      en_fr_tokenizer=en_fr_tokenizer, en_fr_model=en_fr_model,
                      fr_en_tokenizer=fr_en_tokenizer, fr_en_model=fr_en_model)

    print("🎉 Back translation complete.")
