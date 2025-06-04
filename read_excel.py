import pandas as pd

def get_only_triples(text):
    text = text.replace(': [', ':[') 
    if "{" in text and "}" in text:
            start_index = text.rindex('{')
            end_index = text.rindex('}') + 1
            triple_only_text = text[start_index:end_index]
            if ":[" in triple_only_text and ']' in triple_only_text:
                source_sentence = triple_only_text.split(':')[0] 
                source_sentence=source_sentence.split('{')[1]
                words = source_sentence.split() 
                if len(words) <= 2:
                    abbreviation = source_sentence
                else:
                    abbreviation = " ".join(words[:2]) + " ... " + " ".join(words[-2:])
                triple_only_text = triple_only_text.replace(source_sentence, abbreviation)     
    else:
            text = text.replace('\n', '')
            
            if ":[" in text and "]" in text:
                start_index = text.rindex('[')
                end_index = text.rindex(']') + 1
                triple_only_text = text[start_index:end_index]
            else:
                if "[["in text and "]]" in text:
                    start_index = text.rindex('[[')
                    end_index = text.rindex(']]') + 1
                    triple_only_text = text[start_index:end_index]
                else:
                    triple_only_text = text
    return triple_only_text


df = pd.read_excel('result.xlsx')

for i, row in df.iterrows():
    #print(i, ':', row['single_article'], '\nLong Mem: ', row['longmem'], '\n')
    response = row['longmem'].replace('SUBJECT:,', '').replace('RELATION:,', '').replace('OBJECT:,', '')
    response = response.replace('SUBJECT:', '').replace('RELATION:', '').replace('OBJECT:', '')
    response = response.replace('SUBJECT', '').replace('RELATION', '').replace('OBJECT', '')
    print('Processed longmem: ',get_only_triples(response))

print('END!!')
