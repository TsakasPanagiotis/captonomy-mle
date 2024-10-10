# Machine Learning Engineer Assignment: Document Classification and Information Extraction

## Setup

Move inside the project directory  
Create virtual environment `python -m venv .venv`  
Activate virtual environment `.\.venv\Scripts\activate`  
Install requirements `pip install -r requirements.txt`  

## Files and Results

- manual annotations in `sample-ground-truth.csv`

`process-data.ipynb`
- extraction of title, authors, abstract, keywords and introduction
- stores `papers.pkl` pickle file with list of papers in 

`similarity.ipynb`
- sentence similarity with simple and extended (-ext) categories
- stores `similarity-preds.csv` and `similarity-preds-ext.csv` files with paper title, authors, and predicted category
- stores `similarity-preds.json` and `similarity-preds-ext.json` files with structure specified in the instructions

`generation.ipynb`
- text generation with causal language modeling (-clm) and instruction-tuned language modeling (-itlm)
- stores `generation-clm.csv` and `generation-itlm.csv` files with paper title, authors, and predicted category
- stores `generation-clm.json` and `generation-itlm.json` files with structure specified in the instructions

`evaluation.ipynb`
- evaluation of all four approaches: two sentence similarity and two text generation approaches 
- stores `similarity-confusion-matrix.png` and `similarity-confusion-matrix-ext.png` images for the two similarity approaches

## Information Extraction

Related code in `process-data.ipynb`.

The `pypdf` library was used to read the PDF files. The function `extract_text` had two possible extraction modes: plain and layout. First, the plain mode was used to extract the raw text of the first page. Then, regular expressions were used to split the text into the header (title, authors, affiliations), abstract, and perhaps keywords and introduction if they existed. Following, the layout mode was used to separate the title from the authors and their affiliations. Afterward, several heuristic regular expressions were used to find the word `and` and the newline character `\n` around the last author and remove the rest of the following words i.e. the affiliations. Finally, excess white space, redundant commas, leftover characters, and non-alphabetic characters (e.g. indices and IDs) were removed from the authors as cleanup.

## Document Classification

With a sample of 148 papers in `ml-engineer\ICDAR2024_papers.zip`, training or even fine-tuning a model was not an option. It was suggested to use pre-trained transformers from the HuggingFace models. Two approaches were explored: sentence similarity and text generation.

### Sentence Similarity

Related code in `similarity.ipynb`.

The goal was to convert the title and the abstract of each paper into one dense embedding. The same was done to produce one embedding for each of the suggested categories. Then, the best category was selected based on the highest cosine similarity between the categories and the paper embeddings. The `sentence-transformers/all-MiniLM-L6-v2` model was selected (23 million parameters, 100 MBytes). However, one possible issue with this approach was that the name of each category would be converted to an embedding that, in theory, might not fully reflect the essence of the category. Therefore, each category was extended to include descriptions and keywords relevant to the category.

### Text Generation

Related code in `generation.ipynb`.

The goal was to convert the classification task into text generation by giving a model the title, the abstract, and the categories paired with a number from 0 to 6. Then, the model was asked to give the number of the best category for the given paper. The `facebook/opt-125m` model was selected as a well-known lightweight option (125 million parameters, 300 MBytes). However, such small models usually need some fine-tuning on a specific task. This was not possible for this project, but there are larger fine-tuned models available. The `Qwen/Qwen2.5-1.5B-Instruct` model was selected based on popularity (1.5 billion parameters, 3.1 GBytes). This model received as input the same prompt as the other, modified to look like system instruction and user request.

## Evaluation

Related code in `evaluation.ipynb`.

### Titles and Authors

The provided CSV file `ICDAR 2024 paper list.csv` was used as ground truth to compare the extracted titles and authors. From the total of 148 papers, 108 titles were extracted correctly (73\%), 61 times all authors were extracted perfectly (41\%), and 112 times the corresponding author was successfully extracted (75\%). Based on the last two percentages, it is possible that more individual authors are accurately extracted.

### Categories

A small set of 17 manual annotations was made in file `sample-ground-truth.csv` to evaluate the category predictions with two or three samples per category.

#### Sentence Similarity

The predictions using the simple category names and the extended category names had similar distributions overall and a diagonal-centered confusion matrix for the manual annotations. However, only the extended categories produced any "Others" predictions.

| **Category**                     | **Simple** | **Extended** |
|----------------------------------|------------|--------------|
| Tables                           | 10         | 11           |
| Classification                   | 10         | 7            |
| Key Information Extraction       | 22         | 18           |
| Optical Character Recognition    | 75         | 70           |
| Datasets                         | 12         | 3            |
| Document Layout Understanding    | 19         | 18           |
| Others                           | 0          | 21           |


#### Text Generation

Both text-generation models (OPT and instruction-tuned Qwen) failed to predict the categories correctly. OPT classified 147 papers in the "Tables" and 1 in the "Classification" categories, while Qwen classified 116 papers in the "Document Layout Understanding" and 32 in the "Classification" categories. This result could be evidence of the scaling law of large language models. The way these models were used resembles a zero-shot task, which is an emergent ability of much larger language models.


## Conclusion

The information extraction was a difficult task, particularly when there were no standard patterns in the text alone. When the layout of the document needed to be taken into account, the extraction became unpredictable.  

The time and resource constraints of this project reduced the available options for document classification. Sentence similarity with encoder models, especially with extended category descriptions, gave the only satisfactory results and could be further explored with more annotated papers. On the other hand, text generation with decoder models (even with 1 billion parameters) failed the task, indicating that fine-tuning was necessary.
