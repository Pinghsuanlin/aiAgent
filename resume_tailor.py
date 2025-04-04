# resume_tailor.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load resume and jd files and store as strings
with open("resume.txt", "r") as f:
    resume = f.read()

with open("jd.txt", "r") as f:
    jd = f.read()

# Split resume into bullet points
resume_bullets = [line.strip("- ").strip() for line in resume.strip().split("\n") if line.strip()]

# Compare each bullet to the job description
vectorizer = TfidfVectorizer() #gives us a vector for each sentence based on word importance
scores = [] #create empty list to store scores

# core comparison loop
for bullet in resume_bullets:
    vectors = vectorizer.fit_transform([bullet, jd]) # convert into vector
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] #measures the angle between 2 vectors (1 as perfect match; 0 as no similarity)
    scores.append((bullet, score))

# sort values by relevance score (descending)
scores.sort(key=lambda x: x[1], reverse=True)

# Show results
print("🔍 Resume bullets ranked by relevance to job description:\n")
for i, (bullet, score) in enumerate(scores):
    print(f"{i+1}. [{score:.2f}] {bullet}")
