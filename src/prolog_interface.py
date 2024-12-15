from pyswip import Prolog

class PrologInterface:
    def __init__(self, kb_file):
        self.kb_file = kb_file

    def query_prolog(self, query_str="product_info(E,F,C,Su,P,Sa,Cl)"):
        prolog = Prolog()
        prolog.consult(self.kb_file)
        results = list(prolog.query(query_str))
        for r in results:
            print(r)
