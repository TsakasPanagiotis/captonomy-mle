import json
from dataclasses import dataclass

import pandas as pd


categories = [
    'Tables', 
    'Classification', 
    'Key Information Extraction',
    'Optical Character Recognition', 
    'Datasets', 
    'Document Layout Understanding', 
    'Others'
]


@dataclass
class Paper:
    filename: str
    title: str = ''
    authors: str = ''
    abstract: str = ''
    keywords: str = ''
    introduction: str = ''
    
    def __repr__(self):
        return f' filename \n----------\n {self.filename}' + \
               f'\n\n title \n----------\n {self.title}' + \
               f'\n\n authors \n----------\n {self.authors}' + \
               f'\n\n abstract \n----------\n {self.abstract}' + \
               f'\n\n keywords \n----------\n {self.keywords}' + \
               f'\n\n introduction \n----------\n {self.introduction}'


def to_camel_case(text: str) -> str:
    '''Convert a text to camel case'''

    # Split the text into words
    words = text.split()
    
    # Capitalize the first letter of each word except the first word
    camel_case_words = [words[0].lower()] + [word.capitalize() for word in words[1:]]
    
    # Join the words back together without spaces
    camel_case_text = ''.join(camel_case_words)
    
    return camel_case_text


def authors_to_list(authors: str) -> list[str]:
    '''Split the authors string into a list of authors'''

    split = authors.rsplit(' and ', maxsplit=1) # there should be at most one ' and '

    authors_list = split[0].split(',')
    authors_list = [author.strip() for author in authors_list]

    if len(split) == 2: # there should be a last author as inteded after the ' and '
        last_author = split[1].strip()
        authors_list.append(last_author)

    return authors_list


def save_to_csv(papers: list[Paper], predictions: list[str], csv_filename: str) -> None:
    '''Save predictions in a CSV file for further analysis'''

    rows = []
    
    for paper, prediction in zip(papers, predictions):
        rows.append({'filename': paper.filename, 'title': paper.title, 'authors': paper.authors, 'category': prediction})

    predictions_df = pd.DataFrame(rows)

    predictions_df.to_csv(csv_filename, sep=';', index=False)


def save_to_json(papers: list[Paper], predictions: list[str], csv_filename: str) -> None:
    '''Save predictions in a json file with the specified format'''

    result = {to_camel_case(category): [] for category in categories}

    for paper, prediction in zip(papers, predictions):
        result[to_camel_case(prediction)].append({"originalFileName": paper.filename, "title": paper.title, "authors": authors_to_list(paper.authors)})

    with open(csv_filename, 'w') as f:
        json.dump(result, f, indent=4)