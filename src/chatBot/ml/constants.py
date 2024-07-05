data_path = 'data\processed\chatbot\input.json'
tag_col = 'tag'
text_col = 'patterns'
res_col = 'responses'

lr_model_name = 'lr'
svc_model_name = 'svc'

lr_model = 'models\chatbot\ml\lr.pkl'
svc_model = 'models\chatbot\ml\svc.pkl'
vectorizer_filename='models\chatbot\ml\\vectorizer.joblib'

lr_learning_curve = 'src\chatBot\ml\\results\lr_learning_curve.png'
svc_learning_curve = 'src\chatBot\ml\\results\svc_learning_curve.png'

lr_confusion_matrix = 'src\chatBot\ml\\results\lr_confusion_matrix.png'
svc_confusion_matrix = 'src\chatBot\ml\\results\svc_confusion_matrix.png'

classification_report = 'src\chatBot\ml\\results\classification_report.png'
