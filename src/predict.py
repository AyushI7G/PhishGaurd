import torch
from transformers import BertTokenizer, BertForSequenceClassification

email_model = BertForSequenceClassification.from_pretrained("models/bert_email")
tokenizer = BertTokenizer.from_pretrained("models/bert_tokenizer")

email_model.eval()

suspicious_words = ["verify","account","bank","urgent","password","click","login"]

def highlight_text(text):
    for word in suspicious_words:
        text = text.replace(word, f"<span class='highlight'>{word}</span>")
    return text

def check_sender(sender):
    if "@" not in sender:
        return "⚠️ Invalid email"

    if any(x in sender.lower() for x in ["verify","secure","login"]):
        return "⚠️ Suspicious sender"

    return "✅ Sender looks safe"

def predict_email(text, sender):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = email_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    score = probs[0][1].item()

    label = "Phishing" if score > 0.5 else "Safe"

    highlighted = highlight_text(text)
    sender_msg = check_sender(sender)

    return label, score, highlighted, sender_msg