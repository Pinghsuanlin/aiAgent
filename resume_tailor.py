# resume_tailor.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load resume and job description
with open("resume.txt", "r") as f:
    resume = f.read()

with open("job_description.txt", "r") as f:
    jd = f.read()

# Split resume into bullet points
resume_bullets = [line.strip("- ").strip() for line in resume.strip().split("\n") if line.strip()]

# Compare each bullet to the job description
vectorizer = TfidfVectorizer()
scores = []

for bullet in resume_bullets:
    vectors = vectorizer.fit_transform([bullet, jd])
    score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    scores.append((bullet, score))

# Sort by similarity
scores.sort(key=lambda x: x[1], reverse=True)

# Show results
print("🔍 Resume bullets ranked by relevance to job description:\n")
for i, (bullet, score) in enumerate(scores):
    print(f"{i+1}. [{score:.2f}] {bullet}")
