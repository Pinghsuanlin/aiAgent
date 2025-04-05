# resume_tailor.py
# TF-IDF, cosine similarity for meaning-based mataching

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import streamlit as st 

# (unused for streamlit app)
# def load_file(path):
#     with open(path, "r") as f:
#         return f.read().strip()

# turn bullet into a clean string in a list
    # return [line.strip("-* ").rstrip(",.") for line in text.strip().split("\n") if line.strip()]
def extract_bullet(text):
    return [re.sub(r"^[\-\*\s]+", "", line).rstrip(",.").strip()
            for line in text.split("\n") if line.strip()]
    

def rank_bullet(bullets, jd):
    vectorizer = TfidfVectorizer() #gives us a vector for each sentence based on word importance
    scores = [] #create empty list to store scores
    
    for bullet in bullets:
        vectors = vectorizer.fit_transform([bullet, jd]) # convert into vector
        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] #measures the angle between 2 vectors (1 as perfect match; 0 as no similarity)
        scores.append((bullet, score))

    return sorted(scores, key=lambda x: x[1], reverse=True) # sort values by relevance score (descending)

# --- quick testing --- #
# def print_results(scores):
#     print("Resume bullets ranked by relevance to job description:\n")
#     for i, (bullet, score) in enumerate(scores):
#         print(f"{i+1}. [{score:.2f}] {bullet}")

    
# if __name__ == "__main__":
#     resume = load_file("resume.txt")
#     jd = load_file("jd.txt")
#     bullets = extract_bullet(resume)
#     scores = rank_bullet(bullets, jd)
#     print_results(scores)

# --- streamlit UI --- #
st.title("Resume Tailoring Assistant")
st.write("Paste your resume bullets and job description to see how well they match.")

resume_text = st.text_area("Your Resume (bullet points)", height=200)
jd_text = st.text_area("Job Description", height=200)

if st.button("Match Bullets"):
    if resume_text and jd_text:
        bullets = extract_bullet(resume_text)
        results = rank_bullet(bullets, jd_text)
        
        st.subheader("Bullet Points Ranked by Relevance")
        for bullet, score in results:
            st.markdown(f"**Score**: `{score:.2f}` \n {bullet}")
        else:
            st.warning('Please provide both resume and job description.')
