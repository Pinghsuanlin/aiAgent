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
    * `TF-IDF` (term frequency-inverse document frequency) convert text into numbers (ie. numeric vectors)
        * Term frequency (TF): how often a word appears in a document
        * Inverse document frequency (IDF): how unique that word is acorss focuments
        * TF-IDF gives high scores to words that are frequent in the current text (important), and rare in the rest of the corpus (distinctive)
    * `cosine similarity` compare the similarity between sentences (after converting them into vectors (arrays of numbers), from 0 (not similar; large angle) to 1 (identical direction; small angle)
    * `spaCy` is a powerful NLP library used for (more advanced than TF-IDF):
        * tokenization, lemmatization (eg. running -> run)
        * named entity recognition (NER) like job titles, skills, org names
        * part-of-speech tagging like verb, nouns
        * text similarity and vectorization
        * dependency parsing: understand sentence structure
    * API
* Modular format: more readable and maintainable

### Tools:
* Python + `spacy`, `sklearn`
* `streamlit` with small UI


| Task | TF-IDF | spaCy | sentence-transformers (BERT)|
|:-----|:-------|:------|:----------------------|
| raw text comparison | ✅ good | ✅ great (w/vectors)|
| grammar awareness | ❌| ✅ |
| extract entity | ❌ | ✅ |
| keyword overlap | ❌ | ✅ |
| deep sentence understanding | ❌ | ✅ (w embedding)| handle meanings well, even synonyms |
| speed | fast | fast | slower | 
| best for | simple keyword overlap | smart token matching | true semantic similarity | 

