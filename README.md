# Atelier Préparation de Données Tabulaires : bâtiments intelligents (IoT)

Transformer des données brutes de capteurs en un jeu de données **propre et prêt pour le Machine Learning**, avec un pipeline de preprocessing Scikit-learn sans fuite de données.

## Contexte

Une entreprise exploite plusieurs bâtiments intelligents équipés de capteurs IoT. Chaque capteur mesure régulièrement la température, l'humidité, la qualité de l'air (CO₂), la consommation énergétique et le nombre de personnes présentes, et enregistre le mode de fonctionnement et l'état du système de climatisation. Ces données doivent alimenter un modèle capable de **prédire les situations d'alerte** (variable cible `alerte`).

Le fichier brut présente volontairement des problèmes de qualité : valeurs manquantes, doublons, valeurs aberrantes ou impossibles, types incorrects, catégories mal écrites, classes déséquilibrées et échelles très différentes entre variables.

## Objectifs

- Explorer un jeu de données tabulaire et identifier les types de variables
- Détecter et traiter les problèmes de qualité (incohérences, manquants, doublons, aberrants)
- Analyser le déséquilibre de la cible et les corrélations
- Choisir et appliquer un encodage adapté à chaque variable catégorielle
- Normaliser / standardiser les variables numériques
- Éviter la fuite de données (*data leakage*)
- Construire un pipeline de preprocessing avec Scikit-learn
- Produire un dataset final exploitable par un algorithme de Machine Learning

## Structure du projet

```
atelier_prepa_donnees_tab/
├── data/
│   └── smart_building_raw.csv                 # données brutes (jamais modifiées)
├── notebooks/
│   └── atelier_prepa_donnees_tab.ipynb        # notebook de l'atelier
├── exports/
│   ├── smart_building_cleaned.csv             # dataframe nettoyé (Partie 7)
│   ├── smart_building_train_prepared.csv      # jeu d'entraînement préparé
│   ├── smart_building_test_prepared.csv       # jeu de test préparé
│   └── pipeline_alerte.joblib                 # pipeline entraîné (bonus)
└── README.md
```

## Les données

`smart_building_raw.csv` contient **507 observations** et **14 variables** :

| Rôle | Variable | Description |
|---|---|---|
| Identifiant | `id_mesure` | Identifiant de la mesure |
| Date | `date` | Horodatage de la mesure |
| Catégorielle | `batiment` | Identifiant du bâtiment |
| Catégorielle | `type_batiment` | Type de bâtiment (bureau, entrepôt, centre commercial, université…) |
| Catégorielle | `zone` | Zone du bâtiment |
| Numérique | `temperature` | Température (°C) |
| Numérique | `humidite` | Humidité relative (%) |
| Numérique | `co2` | Concentration en CO₂ (ppm) |
| Numérique | `occupation` | Nombre de personnes présentes |
| Numérique | `consommation_kwh` | Consommation énergétique (kWh) |
| Catégorielle | `mode_climatisation` | Mode de fonctionnement de la climatisation |
| Catégorielle | `etat_systeme` | État du système |
| Catégorielle | `jour_semaine` | Jour de la semaine |
| **Cible** | `alerte` | Situation d'alerte (`Oui` / `Non`) |

## Installation et exécution

```bash
git clone <url-du-depot>
cd atelier_prepa_donnees_tab

python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate

pip install pandas numpy matplotlib seaborn scikit-learn jupyter joblib
jupyter notebook notebooks/atelier_prepa_donnees_tab.ipynb
```

- **Scikit-learn ≥ 1.2** est requis (paramètres `sparse_output` et `min_frequency` de `OneHotEncoder`).
- Le notebook lit `../data/` et écrit dans `../exports/` : il doit être lancé depuis le dossier `notebooks/` (comportement par défaut de Jupyter).
- Pour un résultat reproductible, exécuter le notebook de haut en bas (*Restart & Run All*).

## Contenu du notebook

| Partie | Contenu |
|---|---|
| **1. Exploration** | Chargement, dimensions, types de variables, statistiques descriptives, incohérences (humidité hors [0, 100], températures et CO₂ extrêmes, valeurs négatives impossibles, catégories mal orthographiées), valeurs manquantes, doublons, distributions et valeurs aberrantes, fréquences des catégories, déséquilibre de `alerte`, matrice de corrélation |
| **2. Séparation X / y** | Cible `alerte`, choix argumenté des variables explicatives |
| **3. Train / Test** | 80 % / 20 %, `random_state=42`, découpage stratifié sur `alerte` |
| **4. Encodage** | One-Hot des variables catégorielles nominales, justification variable par variable |
| **5. Mise à l'échelle** | Justification (poids des variables dans les distances), `StandardScaler`, pourquoi ne pas calculer les statistiques sur le jeu complet |
| **6. Pipeline** | `ColumnTransformer` : imputation médiane + `StandardScaler` (numériques), imputation mode + `OneHotEncoder` (catégorielles), puis modèle |
| **7. Exportation** | Dataframe nettoyé et jeux préparés dans `exports/` |
| **8. Bonus** | Variables temporelles, écrêtage des extrêmes (`Winsorizer`), comparaison de modèles en validation croisée, sauvegarde du pipeline |

## Démarche et choix méthodologiques

- **Erreur ou valeur extrême réelle ?** Seules les valeurs impossibles ou irrécupérables (humidité négative ou supérieure à 100 %, températures aberrantes, consommation négative…) sont transformées en `NaN`. Les extrêmes plausibles sont conservés : l'objectif étant de détecter des situations anormales, les supprimer détruirait le signal.
- **Catégories** : suppression des espaces puis uniformisation de la casse, avant toute analyse de fréquences ou recherche de doublons.
- **Doublons** : seuls les doublons strictement identiques sont supprimés ; un même `id_mesure` avec des valeurs différentes est un conflit à examiner, pas un doublon.
- **Valeurs manquantes** : taux faibles (moins de 5 % par colonne), donc aucune colonne supprimée ; imputation par la médiane (numériques) et le mode (catégorielles), **calculés sur le train uniquement, dans le pipeline**. Les `NaN` sont volontairement conservés dans `smart_building_cleaned.csv`.
- **Variables explicatives** : `id_mesure` (identifiant), `date` (brute) et `batiment` (identifiant, redondant avec `type_batiment`) sont exclues de X.
- **Découpage** : stratifié, pour garder la même proportion d'alertes en train et en test.
- **Fuite de données** : tout ce qui apprend quelque chose des données (médiane, mode, moyenne, écart-type, catégories) est ajusté (`fit`) sur le train, puis appliqué (`transform`) au train et au test.

## Pipeline de preprocessing

```
Dataset brut
     │
     ▼
Séparation X / y
     │
     ▼
Train / Test (80 / 20, stratifié)
     │
┌────┴───────────────────────┐
▼                            ▼
Variables numériques         Variables catégorielles
     │                            │
     ▼                            ▼
Imputation médiane           Imputation mode
     │                            │
     ▼                            ▼
StandardScaler               OneHotEncoder
     │                            │
     └─────────────┬──────────────┘
                   ▼
            Dataset préparé
                   │
                   ▼
        Modèle de Machine Learning
```

## Résultats clés

| Indicateur | Résultat |
|---|---|
| Taille du jeu brut | 507 observations, 14 variables |
| Répartition de `alerte` | 71,2 % `Non` / 28,8 % `Oui` (déséquilibre modéré, ratio 2,5) |
| Manquants avant nettoyage | Au maximum 4,14 % par colonne (`humidite`), après correction des valeurs impossibles |
| Corrélations marquées | `occupation` et `consommation_kwh` (r = 0,72) ; `temperature` et `consommation_kwh` (r = 0,50) |
| Corrélations faibles | `humidite` et `co2` non corrélées aux autres variables numériques |
| Doublons supprimés | *(à compléter)* |
| Décision sur le CO₂ extrême (> 3 000 ppm) | *(à compléter : conservé ou transformé en `NaN`, avec justification)* |
| Performance du modèle sur le test (F1 macro, rappel de `Oui`) | *(à compléter)* |

Repère à battre : un modèle qui prédit toujours `Non` obtient 71,2 % d'accuracy sans détecter aucune alerte.

## Limites et pistes d'amélioration

- **Petit jeu de données** (507 lignes) : les scores en validation croisée varient beaucoup, et de faibles écarts entre modèles ne sont pas significatifs.
- **Données temporelles** : le découpage aléatoire est demandé par l'énoncé ; pour un système en production, un découpage chronologique (entraîner sur le passé, tester sur le futur) est plus réaliste.
- **Fuite de cible possible** : si `alerte` est calculée à partir d'une des variables explicatives, les scores sont artificiellement bons. À vérifier auprès de la source des données.
- **Pistes** : encodage cyclique du jour de la semaine, rééchantillonnage (SMOTE, sur le train uniquement) si le déséquilibre augmente, optimisation des hyperparamètres.

## Suivi du travail

Un commit par tâche accomplie, avec un message explicite, par exemple :

```
Partie 1.12 : détection et correction des incohérences
Partie 3 : découpage train/test stratifié
Partie 6 : pipeline de preprocessing Scikit-learn
Bonus : winsorisation et comparaison de modèles en validation croisée
```

## Auteur

Rokhaya Coumba DIOUF : atelier réalisé dans le cadre de la formation IA,ODC.

Les données sont fournies dans le cadre de l'atelier.
