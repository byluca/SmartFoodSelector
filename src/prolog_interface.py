# src/prolog_interface.py
from pyswip import Prolog

def query_prolog():
    prolog = Prolog()
    prolog.consult("prolog/knowledge_base.pl")
    # Esegui una query esempio:
    # canzoni_info(Nome, Autore, Cluster)
    # Supponiamo di avere definito dei fatti "song(...)" e "clustered_song(...)"
    # qui mostriamo come fare una query:
    result = list(prolog.query("canzoni_info(N,C,Cl)"))
    for r in result:
        print(r)
