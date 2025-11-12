# CONTEXTE ET OBJECTIF
Je dois créer un système industriel pour comparer 9000 jobs DataStage (fichiers DSX/XML) 
en minimisant la consommation de tokens Claude AI et en maximisant les performances.

# ARCHITECTURE CIBLE
Pipeline en 6 phases:
1. Extraction de fingerprints structurels (parsing XML local, 0 tokens)
2. Clustering structurel par hash de signature (0 tokens)
3. Clustering sémantique avec embeddings locaux (sentence-transformers, 0 tokens)
4. Sélection de représentants (~1000 jobs sur 9000)
5. Comparaison fine par batch avec Claude API + prompt caching
6. Propagation des résultats et génération de rapport interactif

# CONTRAINTES TECHNIQUES
- Python 3.10+
- Traitement asynchrone (asyncio) pour I/O
- Multiprocessing pour parsing XML
- Cache Redis pour éviter recomparaisons
- Budget token: max 100M tokens (~$300)
- Performance: traiter 9000 jobs en < 2h