import joblib
import pandas as pd
import numpy as np
import re
import os

def extract_features_from_text(text):
    """
    Extracts the 57 features required by the Spambase model from a raw text string.
    This function replicates the feature set from the UCI Spambase dataset.
    """
    # Hardcoded feature names in the correct order (cleaned for XGBoost)
    feature_columns = [
        'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
        'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet',
        'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
        'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
        'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
        'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money',
        'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650',
        'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857',
        'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology',
        'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
        'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project',
        'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference',
        'char_freq_;', 'char_freq_(', 'char_freq__', 'char_freq_!',
        'char_freq_$', 'char_freq_#', 'capital_run_length_average',
        'capital_run_length_longest', 'capital_run_length_total'
    ]

    # Initialize a dictionary to hold feature values
    features = {col: 0 for col in feature_columns}
    
    # --- Pre-processing for feature calculation ---
    text_lower = text.lower()
    # Find all sequences of words (alphanumeric strings)
    words = re.findall(r'[a-zA-Z0-9]+', text_lower)
    total_words = len(words) if len(words) > 0 else 1
    total_chars = len(text) if len(text) > 0 else 1

    # --- 1. Calculate Word Frequencies ---
    word_freq_cols = [col for col in feature_columns if 'word_freq' in col]
    for col in word_freq_cols:
        word = col.replace('word_freq_', '')
        features[col] = (words.count(word) / total_words) * 100

    # --- 2. Calculate Character Frequencies ---
    char_freq_cols = [col for col in feature_columns if 'char_freq' in col]
    for col in char_freq_cols:
        char = col.replace('char_freq_', '')
        # Special case for cleaned '[' character
        if char == '_':
            char = '['
        features[col] = (text.count(char) / total_chars) * 100
        
    # --- 3. Calculate Capital Run Length Features ---
    # Find all sequences of 1 or more uppercase letters
    capital_runs = re.findall(r'[A-Z]+', text)
    if capital_runs:
        run_lengths = [len(run) for run in capital_runs]
        features['capital_run_length_average'] = np.mean(run_lengths)
        features['capital_run_length_longest'] = np.max(run_lengths)
        features['capital_run_length_total'] = np.sum(run_lengths)
    else:
        features['capital_run_length_average'] = 0
        features['capital_run_length_longest'] = 0
        features['capital_run_length_total'] = 0

    # Create a DataFrame with the correct column order
    return pd.DataFrame([features], columns=feature_columns)


def main():
    """
    Main function to load the model and run the interactive checker.
    """
   
    model_path = 'models/best_spam_classifier.joblib'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        print("Please make sure you have run the model training notebook first.")
        return
        
    model = joblib.load(model_path)
    print("Spam classification model loaded successfully.")
    
    #Interactive Loop
    print("\n--- Interactive Spam Checker ---")
    print("Enter your email text below. Type 'ENDEMAIL' on a new line when you're done.")
    print("Type 'quit' to exit.")

    while True:
        lines = []
        while True:
            line = input()
            if line.strip().lower() == 'endemail':
                break
            if line.strip().lower() == 'quit':
                print("Exiting. Goodbye!")
                return
            lines.append(line)
        
        email_text = "\n".join(lines)
        
        if not email_text.strip():
            print("Input is empty. Please try again.")
            continue

        # Extract features and make a prediction
        features_df = extract_features_from_text(email_text)
        prediction = model.predict(features_df)[0]
        prediction_proba = model.predict_proba(features_df)[0]

        print("\n--- Prediction ---")
        if prediction == 1:
            print(f" Result: This looks like SPAM.")
            print(f"   Confidence: {prediction_proba[1]*100:.2f}%")
        else:
            print(f"👍 Result: This looks like NOT SPAM (Ham).")
            print(f"   Confidence: {prediction_proba[0]*100:.2f}%")
        print("--------------------")
        print("\nEnter another email or type 'quit' to exit.")


if __name__ == '__main__':
    main()