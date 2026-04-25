# Install libraries
!pip install sentence-transformers PyPDF2 transformers torch

from google.colab import files
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch

# Upload PDF
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

# Read PDF
reader = PdfReader(file_name)
doc = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        doc += text

print("PDF loaded successfully!")

# Load models
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

ai_detector = pipeline(
    "text-classification",
    model="roberta-base-openai-detector"
)

# Split sentences
def split_sentences(text):
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    return [s.strip() for s in sentences if len(s.strip()) > 20]

sentences = split_sentences(doc)

print("Total sentences:", len(sentences))

# Encode sentences
embeddings = similarity_model.encode(
    sentences,
    convert_to_tensor=True
)

threshold = 0.75

match_count = 0
ai_scores = []

print("\n======== ANALYSIS START ========\n")

for i in range(len(sentences)):
    
    # AI detection
    result = ai_detector(sentences[i])[0]
    ai_score = result['score']
    
    ai_scores.append(ai_score)

    for j in range(i + 1, len(sentences)):
        sim = util.cos_sim(
            embeddings[i],
            embeddings[j]
        ).item()

        if sim > threshold:

            match_count += 1

            print("Match:", match_count)
            print("Sentence:", sentences[i][:120])
            print("Similarity:", round(sim * 100, 2), "%")
            print("AI Score:", round(ai_score * 100, 2), "%")
            print("----------------------------")

# Final metrics

avg_ai_score = sum(ai_scores) / len(ai_scores)

similarity_index = (
    match_count / len(sentences)
) * 100

# Risk level

if avg_ai_score > 0.7:
    risk = "HIGH"
elif avg_ai_score > 0.4:
    risk = "MEDIUM"
else:
    risk = "LOW"

print("\n======== FINAL REPORT ========\n")

print("Total Sentences:", len(sentences))
print("Similarity Index:", round(similarity_index, 2), "%")
print("AI Probability:", round(avg_ai_score * 100, 2), "%")
print("Risk Level:", risk)

# Save report

with open("Turnitin_style_report.txt", "w") as f:

    f.write("DOCUMENT ANALYSIS REPORT\n\n")
    f.write("Total Sentences: " +
            str(len(sentences)) + "\n")

    f.write("Similarity Index: " +
            str(round(similarity_index, 2)) + "%\n")

    f.write("AI Probability: " +
            str(round(avg_ai_score * 100, 2)) + "%\n")

    f.write("Risk Level: " +
            risk)

print("\nReport saved as report.txt")
