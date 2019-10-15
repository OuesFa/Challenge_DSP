# Challenge_DSP

**Objectif:** Estimer le CA

## I. La démarche:

Nous avons souhaité réaliser une étude qui suit le cheminement d'une démarche d'étude  : à comprendre acquérir la donnée, l'observer, se l'approprier et la nettoyer. Suite à cela nous appliquons nos algorithmes de machine learning afin de prédire notre cible. Néanmoins le chemin est plus ardu qu'il n'y parait : il est nécessaire pour chaque model de paramétrer l'algorithme choisi afin d'en tirer les meilleurs performances, mais très vite on s'aperçoit que les résultats atteignent rapidement un plafond que l'on ne peut améliorer seulement en retravaillant le tuning de l'algorithme.

Le gain le plus important sur les performances est atteint en retraitant la donnée la plus intelligemment possible, dans notre situation, avec des retraitements simples on obtient des gains de performances notables tant bien pour la métrique RMSE de prédilection que pour d'autres métriques (tel que MAPE).

## II. Data management:

- Régression sur les valeurs manquantes
- Imputation des données catégorielles par leur top représentant
- Encodage disjonctif complet (dummy) sur les variables catégorielles
- Traitement des outliers


## III. Choix des algorithmes:

- 1er. Régression Linéaire Lasso sur données polynomiales d'ordre 2
- 2nd. Régression Linéaire standard (sans pénalisation) sur données polynomiales d'ordre 2
- 3eme. XGBoost (profondeur 4, nb_estimateurs : 400, learning rate à 0.05) sur donnée d'ordre 1.

# <font color='red'>IMPORTANT : </font> 

- Les notebooks fournis font office de rapport et de code à la fois, il est possible de les exécuter néanmoins les affichages seront réinitialisés et il sera nécessaire d'attendre la fin de l'exécution du notebook.
- L'ordre d'exécution des notebooks est important, il est nécessaire de les exécuter dans l'ordre suivant :
    - Ch1 -Data Exploration _ Préparation.ipynb
    - Ch2 -Apprentissage.ipynb
    - Ch3 -Fine Tuning & Améliorations.ipynb
- Un script d'exécution pure est également fourni pour régénérer le fichier, néanmoins pour les explications il est nécessaire de se référer aux notebooks ci-dessus.
- Il est possible de consulter les notebooks en ligne (affiche légèrement altérée) directement sur mon repo github au lien suivant : https://github.com/dvp-tran/Axa_Challenge
    Il est possible de cloner le repo.
