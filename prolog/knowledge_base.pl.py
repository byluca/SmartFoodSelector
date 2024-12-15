% prolog/knowledge_base.pl
% Esempio semplice
% song(A1,A2,A3,...) fatti con info su canzoni
% clustered_song(...) stessi argomenti e ultima colonna cluster

% Esempio fittizio
song(0.5,0.2,0.7,0.1, 'Autore_A', 'Canzone_X').
clustered_song(0.5,0.2,0.7,0.1, 2).

canzoni_info(NomeCanzone, Autore, Cluster) :-
    song(A1,A2,A3,A4,Autore,NomeCanzone),
    clustered_song(A1,A2,A3,A4,Cluster).
