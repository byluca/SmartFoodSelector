# src/prolog_interface.py
from pyswip import Prolog

def query_prolog():
    prolog = Prolog()
    prolog.consult("prolog/knowledge_base.pl")
    # Esegui una query di esempio:
    # product_info(E,F,C,Su,P,Sa,Cl)
    # Otteniamo tutti i fatti
    result = list(prolog.query("product_info(E,F,C,Su,P,Sa,Cl)"))
    for r in result:
        # r è un dizionario con chiavi 'E','F','C','Su','P','Sa','Cl'
        print(r)
