import pandas as pd
from compare_model import compute_triple_similarity_roberta

def get_triples(input_longmemstr):
    final_triples = []
    import re
    def extract_content(text):
        text = text.replace("'", "")
        text = text.replace('"', "")
        text = text.replace('\n', "")
        text = text.replace(':  ', "")
        text = text.replace(': ', "")
        text = text.replace(' :', "")
        text = re.sub(r"(SUBJECT:|Subject:|OBJECT:|RELATION:)", "", text)
        pattern = r"\[(.*?,.*?,.*?)\]"
        matches = re.findall(pattern, text)
        return matches

    input_longmem_listsrt = extract_content(input_longmemstr)
    import ast

    for i in input_longmem_listsrt:
        try:
            i = i.replace(',', '","')
            i = '["' + (i) + '"]'
            result = ast.literal_eval(i)
            final_triples.append(result)
        except:
            print('error')
            continue

    for tri in final_triples:
        for i in range(len(tri)):
            tri[i] = tri[i].strip()

    return final_triples


summary_df = []
triples_list = []
for i in range(3):
    df = pd.read_excel(f'result{i}.xlsx')
    for i in range(len(df)):
        triples = get_triples(df['longmem'][i])
        extracted_triples = list()
        for triple in triples:
            if triple:
                extracted_triples.append(triple)

        triples_list.append(extracted_triples)
        new_df = pd.DataFrame({'single_article': [str(df['single_article'][i])], 'longmem': [str(triples)]})
        summary_df.append(new_df)


summary_df = pd.concat(summary_df, ignore_index=True)
summary_df.to_excel('summary.xlsx', index=False)
    
df_truth = pd.read_excel('Knowledge Graph Construction Benchmark And Result.xlsx')
truth_triples = []
for i in range(len(df_truth)):
    triples = get_triples(df_truth['Ground Truth'][i])
    truth_triples.append(triples)

size = len(triples_list)
print(size, len(truth_triples))
total_metrics = dict()
for i in range(size):
    metrics, matches = compute_triple_similarity_roberta(triples_list[i], truth_triples[i], 0.5)
    for metric, value in metrics.items():
        if metric not in total_metrics:
            total_metrics[metric] = value
        else:
            total_metrics[metric] += value


print("Average Metrics:")
for metric, value in total_metrics.items():
    print(f"{metric}: {value/size:.2f}")
