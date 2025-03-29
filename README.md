# Projet Programmation par Contraintes

## Introduction

Ce projet a pour but de vous permettre d'appliquer concrètement les méthodes et outils vus en cours sur les problématiques de recherche (Search), de programmation par contraintes (CSP), et de raisonnement logique avancé (SAT/SMT). Vous serez amenés à résoudre des problèmes réels ou réalistes à l'aide de ces techniques en développant un projet complet, depuis la modélisation jusqu'à la solution opérationnelle.


## Modalités du projet

### Livrables

Chaque groupe devra forker ce dépôt  Git et déposer son travail dans un répertoire dédié du dépôt. Ce répertoire contiendra :

- Le code source complet, opérationnel, documenté et maintenable (en Python, C#, C++, ou autre).
- Le matériel complémentaire utilisé pour le projet (datasets, scripts auxiliaires, etc.).
- Les slides utilisés lors de la présentation finale.
- Un notebook explicatif détaillant les étapes du projet, les choix de modélisation, les expérimentations et les résultats obtenus.

Les livraisons se feront via des **pull requests**, qui devront être régulièrement mises à jour durant toute la durée du projet de sorte que l'enseignant puisse suivre l'avancement et éventuellement apporter des retours et de sorte que tous les élèves aient pu prendre connaissance des travaux des autres groupes avant la soutenance avec évaluation collégiale.

### Présentation

- Présentation orale finale avec support visuel (slides).
- Démonstration de la solution opérationnelle devant la classe.

### Évaluation

- Évaluation collégiale : chaque élève évaluera les autres groupes en complément de l’évaluation réalisée par l’enseignant.
- Critères : clarté, originalité, robustesse de la solution, qualité du code, pertinence des choix méthodologiques et organisation.

## Utilisation des LLMs

### Outils à disposition

Pour faciliter la réalisation du projet, vous aurez accès à plusieurs ressources avancées :

- **Plateforme Open-WebUI** : intégrant des modèles d'intelligence artificielle d'OpenAI et locaux très performants, ainsi que des plugins spécifiques et une base de connaissances complète alimentée par la bibliographie du cours (indexée via ChromaDB, taper # en conversation pour invoquer les KB).
- **Clés d'API OpenAI et locales** : mise à votre disposition pour exploiter pleinement les capacités des modèles GPT dans vos développements.
- **Notebook Agentique** : un notebook interactif permettant d'automatiser la création ou la finalisation de vos propres notebooks, facilitant ainsi la structuration et l'amélioration de vos solutions.

### Combinaison LLM et CSP

Vous avez également la possibilité d'intégrer les Large Language Models (LLMs) directement dans votre projet CSP afin d'en étendre significativement les capacités, via :

- Une utilisation directe des LLM pour assister la conception ou la résolution de CSP complexes.
- Le recours au "function calling" : fournir à un LLM un accès direct à votre CSP, permettant ainsi au modèle de piloter la résolution du problème de manière plus flexible et intuitive. Le notebook agentique fourni constitue un exemple pratique et efficace de cette méthodologie légère mais puissante. La normalisation en cours des MCPs constitue également un excellent exemple d'application de cette approche (vous développez un MCP utilisant la PrCon dans le cadre de votre projet).

## Exemples de Sujets

Ces exemples couvrent des problématiques variées et réelles, illustrant la diversité des domaines applicatifs potentiels (planification de ressources, optimisation logistique, jeux de logique, gestion des plannings, etc.) en lien direct avec les compétences acquises dans ce cours. Vous pouvez choisir l'un de ces sujets ou proposer le vôtre en prenant soin de le faire valider au préalable.

## 18. Optimisation de portefeuille financier sous contraintes  
Sélection d’un ensemble d’investissements (actions, actifs) maximisant le rendement attendu pour un niveau de risque donné, en respectant diverses contraintes (budget maximum, limites par secteur, etc.). Le problème se ramène souvent à une optimisation combinatoire (et continue) où l’on décide quelles fractions du capital allouer à chaque actif. On peut modéliser cela en CP/MILP en introduisant des variables entières (nombre d’unités de chaque actif) ou booléennes (actif sélectionné ou non) avec des contraintes de budget et de diversification, et en maximisant une fonction d’utilité (rendement moins pénalité de risque). Par exemple, un modèle CP-SAT peut utiliser des variables entières pour chaque produit financier (avec bornes min/max d’investissement) et ajouter des contraintes de type knapsack et d’allocation ([algorithm - Portfolio optimization problem: improving the O(n!) solution - Stack Overflow](https://stackoverflow.com/questions/73184010/portfolio-optimization-problem-improving-the-on-solution#:~:text=indicator%20%3D%20solver.BoolVar%28,Solve)), l’objectif étant de maximiser le profit total ([algorithm - Portfolio optimization problem: improving the O(n!) solution - Stack Overflow](https://stackoverflow.com/questions/73184010/portfolio-optimization-problem-improving-the-on-solution#:~:text=3)). Grâce aux solveurs modernes, ce type de problème (combinatoire non linéaire approché linéairement) peut être résolu sur des portefeuilles de taille raisonnable, offrant une solution optimale respectant strictement les contraintes de gestion du risque.  
**Références :** Markowitz (1952), *Portfolio Selection* – modèle moyenne/variance; StackOverflow (2022) – formulation d’un portefeuille en CP-SAT (OR-Tools) ([algorithm - Portfolio optimization problem: improving the O(n!) solution - Stack Overflow](https://stackoverflow.com/questions/73184010/portfolio-optimization-problem-improving-the-on-solution#:~:text=3)) ([algorithm - Portfolio optimization problem: improving the O(n!) solution - Stack Overflow](https://stackoverflow.com/questions/73184010/portfolio-optimization-problem-improving-the-on-solution#:~:text=indicator%20%3D%20solver.BoolVar%28,Solve)); Michalewicz & Fogel (2000), *How to Solve It: Modern Heuristics* – chapitre sur optimisation financière.

Les [publications suivantes](https://drive.google.com/file/d/1KPokq-5Z_aj_T5ysXyqnFebaoefpKU-6/view?usp=sharing) constituent un ensemble de ressources variées introduisant le problème d'optimisation de portefeuille financier sous contrainte et différentes techniques autour de la programmation par contrainte.


