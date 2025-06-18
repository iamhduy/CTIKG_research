from summarize import get_triples
import networkx as nx
import pandas as pd

def build_graph_allsentences(df):
    G = nx.DiGraph()
    for i in range(1, len(df)):
        fulltext=df.loc[i,'single_article']
        longmemstr=df.loc[i,'longmem']
        triples = get_triples(longmemstr)
        for triple in triples:
            source = triple[0]
            target = triple[2]
            G.add_edge(source, target, articletext=fulltext, rel=triple[1])
    return G

df_triples = pd.read_excel('summary.xlsx')
G = build_graph_allsentences(df_triples)
print(G.number_of_nodes(), G.number_of_edges())


