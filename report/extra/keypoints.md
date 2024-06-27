# Punti chiave per la tesi

1. Parte 1
   1. Ad oggi si usano architetture di deep learning per svolgere task legati all'elaborazione del linguaggio, categoria di cui la sentiment analysis fa parte, come mai usare un modello di machine learning?
      1. Volendo indagare su architetture non classiche era ragionevole cercare problemi di apprendimento con una formulazione QUBO naturale, tra questi troviamo le SVM
   2. A livello di risultati i transformer ottengono performance migliori nel task di classificazione, ma il loro addestramento è sicuramente più oneroso dal punto di vista di risorse hardware utilizzate/tempo impiegato per l'addestramento. La riscrittura di questi problemi con tecniche di ML permette una risoluzione del task in tempi decisamente più brevi e con performance non eccessivamente inferiori
2. Parte 2
   1. Analizzando i risultati è stato rilevato uno sbilanciamento che porta il solver ibrido a passare la maggior parte del tempo nella risoluzione classica, è possibile ribilanciare questi tempi sfruttando maggiormente la QPU? (potenzialmente per ottenere tempi ancora minori)
3. Parte 3
   1. Studio del minor embedding
   2. Scomposizione algebrica dei problemi QUBO e come continuare la ricerca per l'implementazione di solver ibridi alternativi
