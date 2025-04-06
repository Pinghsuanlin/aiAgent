# resume_tailor.py
# TF-IDF, cosine similarity for meaning-based mataching

from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import re
import streamlit as st # type: ignore
import PyPDF2 # type: ignore
import docx # type: ignore
import io
import spacy # type: ignore # smart keyword extract
from spacy.lang.en.stop_words import STOP_WORDS # type: ignore
from sentence_transformers import SentenceTransformer, util # type: ignore # BERT
from difflib import SequenceMatcher
from keyword_config import COMMON_JUNK, SOFT_KEYWORDS, HARD_KEYWORDS

nlp = spacy.load('en_core_web_md')
bert_model = SentenceTransformer('all-MiniLM-L6-v2')


# --- (unused for streamlit app) --- #
# def load_file(path):
#     with open(path, "r") as f:
#         return f.read().strip()

# turn bullet into a clean string in a list
    # return [line.strip("-* ").rstrip(",.") for line in text.strip().split("\n") if line.strip()]
def extract_bullet(text):
    return [re.sub(r"^[\-\*\s]+", "", line).rstrip(",.").strip()
            for line in text.split("\n") if line.strip()]


# Add spaCy keyword extraction helper with token matching (option 1)
def extract_keywords(text):
    doc = nlp(text.lower())
    keywords = set()
    
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()
        if len(phrase) <= 2:
            continue
        if all(token.text in STOP_WORDS for token in chunk):
            continue
        if not any(token.is_alpha for token in chunk):
            continue
        if phrase in COMMON_JUNK:
            continue
        keywords.add(phrase)
        
    return keywords

    # lemmatization (option 2)
    # for token in doc:
    #     if token.pos_ in {'NOUN', 'PROPN', 'VERB', 'ADJ'} and not token.is_stop and token.is_alpha:
    #         keywords.add(token.lemma_.lower()) # use lemma for matching


# (option 3)
def keyword_match_score(k, b):
    return SequenceMatcher(None, k.lower(), b.lower()).ratio()

# Add a new function for keyword suggestion
def find_missing_keywords(bullet, jd_keywords, threshold=0.7):
    bullet_keywords = extract_keywords(bullet)
    missing = []
    for k in jd_keywords:
        best_match_score = max((keyword_match_score(k, b) for b in bullet_keywords), default=0)
        if best_match_score < threshold:
            missing.append(k) # jd_keywords - bullet_keywords # exact match
    return missing


def categorize_keywords(keywords):
    soft = []
    hard = []
    other = []
    
    for kw in keywords:
        if any(word in kw for word in SOFT_KEYWORDS):
            soft.append(kw)
        elif any(word in kw for word in HARD_KEYWORDS):
            hard.append(kw)
        else:
            other.append(kw)
    
    return soft, hard, other


# regex cleanup after parsing resume
def clean_resume_text(raw_text):
    lines = raw_text.split('\n')
    clean_lines = []
    for line in lines:
        line = line.strip()
        # skip section headers or boilerpalte
        if re.match(r"^(work experience|eductaion|core competency|technical skills|summary|certificates)$", line.lower()):
            continue
        if re.match(r"^[a-zA-Z\\s]+ at [a-zA-Z\\s]+$", line):
            continue
        if re.search(r"\\b(19|20)\\d{2}\\b", line): #date
            continue
        if len(line) < 3:
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines)

# Named Entity Recognition (NER) to clean resume before scoring
def remove_named_entities(text, labels_to_remove={'PERSON', 'ORG', 'GPE', 'DATE'}):
    doc = nlp(text)
    spans_to_remove = [ent for ent in doc.ents if ent.label_ in labels_to_remove]
    
    cleaned_text = text
    for ent in spans_to_remove:
        cleaned_text = cleaned_text.replace(ent.text, "")
    return cleaned_text

# --- option 1. TFIDF --- #   
# def rank_bullet(bullets, jd):
#     vectorizer = TfidfVectorizer() #gives us a vector for each sentence based on word importance
#     scores = [] #create empty list to store scores
    
#     for bullet in bullets:
#         vectors = vectorizer.fit_transform([bullet, jd]) # convert into vector
#         score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] #measures the angle between 2 vectors (1 as perfect match; 0 as no similarity)
#         scores.append((bullet, score))

#     return sorted(scores, key=lambda x: x[1], reverse=True) # sort values by relevance score (descending)

# --- option 2. BERT --- #
def rank_bullet_bert(bullets, jd):
    jd_embedding = bert_model.encode(jd, convert_to_sensor=True)
    bullet_embeddings = bert_model.encode(bullets, convert_to_sensor=True)
    
    similarities = util.cos_sim(bullet_embeddings, jd_embedding) # use cosine similarity to compute relevance
    results = [(bullets[i], float(similarities[i])) for i in range(len(bullets))]
    return sorted(results, key=lambda x: x[1], reverse=True)


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
st.write("Paste/Upload your resume bullets and job description to see how well they match.")

# option 1. text input
# resume_text = st.text_area("Your Resume (bullet points)", height=200)

# option 2. file upload
st.subheader("Upload Resume or Paste Text")
uploaded_file = st.file_uploader("Upload a pdf or doc resume", type=['pdf', 'docx'])
resume_text = ""

if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1]
    
    if file_type == 'pdf':
        reader = PyPDF2.PdfReader(uploaded_file)
        resume_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file_type == 'docx':
        doc = docx.Document(uploaded_file)
        resume_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        
        reader = PyPDF2.PdfReader(uploaded_file)
        resume_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
else:
    resume_text = st.text_area("Or paste your resume text here", height=200)



jd_text = st.text_area("Job Description", height=200)



if st.button("Match Bullets"):
    if resume_text and jd_text:
        cleaned_resume = remove_named_entities(resume_text)
        resume_text = clean_resume_text(cleaned_resume)
        bullets = extract_bullet(resume_text)
        results = rank_bullet_bert(bullets, jd_text)
        jd_keywords = extract_keywords(jd_text)   
        
        soft_kw, hard_kw, other_kw = categorize_keywords(jd_keywords)
        
        st.subheader("Key Concepts in Job Description")
        if hard_kw:
            st.markdown("**Hard Skills:**")
            st.markdown(", ".join(hard_kw))
        if soft_kw:
            st.markdown("**Soft Skills:**")
            st.markdown(", ".join(soft_kw))
        if other_kw:
            st.markdown("**Other Skills:**")
            st.markdown(", ".join(other_kw))
        
        all_resume_text = " ".join(bullets)
        resume_keywords = extract_keywords(all_resume_text)
        
        overlap = jd_keywords & resume_keywords
        missing =jd_keywords - resume_keywords
        
        st.markdown(f"Your resuem covers **{len(overlap)} / {len(jd_keywords)}** keywords from the job description.")
        if missing:
            st.subheader("Suggested keywords to add")
            soft_miss, hard_miss, other_miss = categorize_keywords(missing)
        
            if hard_miss:
                st.markdown("**Hard Skills:**" + ", ".join(hard_miss[:5]))
            if soft_miss:
                st.markdown("**Soft Skills:**" + ", ".join(soft_miss[:5]))
            if other_miss:
                st.markdown("**Other Skills:**" + ", ".join(other_miss[:5]))
            
            st.markdown("Consider adding: "+", ".join(list(missing)[:10])) #up to 10 missing keywords
                 
        st.subheader("Bullet Points Ranked by Relevance")

        for bullet, score in results:
            
            # missing = find_missing_keywords(bullet, jd_keywords)
            
            st.markdown(f"**Score**: `{score:.2f}` \n {bullet}")
            
            if score < 0.35: # threshold (simple rewrite suggestion)
                suggestion = f"Try adding keywords like : {', '.join(list(missing)[:3])}"
                rewrite = f"• {bullet} - incorporated {', '.join(list(missing)[:2])} to align with the jd"
                st.markdown(f"_Suggestion_: {suggestion}")
                st.markdown(f"_Rewrite Idea_: {rewrite}")
                
            
            # if missing:
            #     clean_missing = [m for m in missing if len(m.split()) > 1 and len(m) > 5]
            #     if clean_missing:
            #         st.markdown(f"_Missing keywords_: `{','.join(missing[:5])}`") #limit to top 5
            #     else:
            #         st.markdown("_no major missing keywords_")
            # else:
            #     st.markdown("_no major missing keywords_")

        st.markdown('---')
