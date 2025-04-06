# Resume Tailoring Bot
## Project Goal: 
Automatically rewrite a resume (or bullet points) to match a specific job description.

Take:
1. a resume: plain text or bullet points, or turn PDF/work to text
2. a job description

Output:
1. matched skills/keywords
2. suggestions to tailor resume content
3. optinoally rephrase bullet resume with help from LLM Transformers

## Techniques used: 
* Keyword extraction: fuzzy matching (token or embedding similarity) instead of exact match
* NLP matching (TF-IDF, cosine similarity, spacy to add missing keyword highlighter)
    * `TF-IDF` (term frequency-inverse document frequency) convert text into numbers (ie. numeric vectors)
        * Term frequency (TF): how often a word appears in a document
        * Inverse document frequency (IDF): how unique that word is acorss focuments
        * TF-IDF gives high scores to words that are frequent in the current text (important), and rare in the rest of the corpus (distinctive)
    * `cosine similarity` compare the similarity between sentences (after converting them into vectors (arrays of numbers), from 0 (not similar; large angle) to 1 (identical direction; small angle)
        * it looks at the direction, not length (that's why it's good to compare text meaning even when length differs)
    * `spaCy` is a powerful NLP library used for (more advanced than TF-IDF): `python -m spacy download en_core_web_md`
        * tokenization (split text into words), lemmatization (eg. running -> run)
        * named entity recognition (NER) eg. job titles, skills, org names
        * part-of-speech tagging like verb, nouns
        * text similarity and vectorization
        * dependency parsing: understand sentence structure
        * word embeddings: semantic similarity 
        * chunking but need to filter out stopwords and generic phrases (eg. extract "time series analysis" as a noun phrase)
    * API
* Modular format: more readable and maintainable


### Tools:
* Python `spacy`, `sklearn`
* `streamlit` with small UI
* `PyPDF2` extract text from PDFs, `python-docx` does the same for word(.docx) files


| Task | TF-IDF | spaCy | sentence-transformers (BERT)|
|:-----|:-------|:------|:----------------------|
| raw text comparison | ✅ good | ✅ great (w/vectors)|
| grammar awareness | ❌| ✅ |
| extract entity | ❌ | ✅ |
| keyword overlap | ❌ | ✅ |
| deep sentence understanding | ❌ | ✅ (w embedding)| handle meanings well, even synonyms |
| speed | fast | fast | slower | 
| best for | simple keyword overlap | smart token matching | true semantic similarity | 


* `spaCy` framework
<br>

| Model | Size | Features | Use Case |
|:------|:---- |:-------- |:-------- |
| `en_core_web_sm` | small | no word vectors (less accurate) | fast, basic use only |
| `en_core_web_md` | medium | has word vectors | recommended for similarity |
| `en_core_web_lg` | large | better accuracy, slower | best for deep analysis |