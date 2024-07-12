data_path = 'data\processed\chatbot\input.json'
model_path = 'models\chatbot\deeplearning\chatbot_model.h5'
text_col = 'patterns'
res_col = 'responses'

preprocessing_steps = [
    'translate_emojis_to_text',
    'lower_sentence',
    'remove_nonascii_diacritic',
    'remove_emails',
    'clean_html',
    'remove_url',
    'replace_repeated_chars',
    'expand_sentence',
    'remove_possessives',
    'remove_extra_space',
    # 'tokenize_sentence',
    # 'remove_stop_words',
    # 'detokenize_sentence'
]
