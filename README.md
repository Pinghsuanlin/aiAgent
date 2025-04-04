# Resume Tailoring Bot
## Project Goal: 
Automatically rewrite a resume (or bullet points) to match a specific job description.

Take:
1. a resume: plain text or bullet points
2. a job description

Output:
1. matched skills/keywords
2. suggestions to tailor resume content
3. optinoally rephrase bullet resume with help from LLM Transformers

## Techniques used: 
* Keyword extraction
* NLP matching (TF-IDF, cosine similarity)
* API

### Tools:
* Python + `spacy`, `sklearn`
* `streamlit` with small UI


`spaCy` is a powerful NLP library used for (more advanced than TF-IDF):
* tokenization, lemmatization
* named entity recognition (NER)
* part-of-speech tagging
* text similarity and vectorization
