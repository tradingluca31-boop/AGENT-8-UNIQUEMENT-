# 📖 GUIDE D'UTILISATION - SYSTÈME DE TRACKING AGENT 8

**Date**: 2025-12-01
**Version**: 1.0

---

## 🎯 OBJECTIF

Ce guide explique comment utiliser le système de tracking automatique des modifications pour le projet AGENT 8. Ce système permet de:

- ✅ Enregistrer toutes les modifications du projet
- ✅ Générer des rapports quotidiens automatiques
- ✅ Faciliter la collaboration entre différents Claude Code agents
- ✅ Maintenir une traçabilité complète du projet

---

## 📂 FICHIERS DU SYSTÈME

### Fichiers Principaux

1. **[ACTUALITE_MISE_A_JOUR.md](../ACTUALITE_MISE_A_JOUR.md)**
   - Document central de l'état du projet
   - Mis à jour automatiquement par le système
   - À consulter AVANT de commencer à travailler

2. **[analysis/modification_tracker.py](../analysis/modification_tracker.py)**
   - Script Python principal du système de tracking
   - CLI complète pour logger et générer des rapports

3. **[docs/MODIFICATIONS_LOG.json](MODIFICATIONS_LOG.json)**
   - Base de données JSON de toutes les modifications
   - Généré et mis à jour automatiquement

4. **[docs/daily_reports/](daily_reports/)**
   - Dossier contenant tous les rapports quotidiens
   - Format: `RAPPORT_QUOTIDIEN_YYYYMMDD.md`

### Fichiers Batch (Windows)

1. **[launchers/LOG_MODIFICATION.bat](../launchers/LOG_MODIFICATION.bat)**
   - Interface interactive pour logger une modification
   - Plus simple que la ligne de commande

2. **[launchers/GENERATE_REPORT.bat](../launchers/GENERATE_REPORT.bat)**
   - Génère le rapport quotidien en un clic

---

## 🚀 DÉMARRAGE RAPIDE

### Option 1: Utiliser les fichiers Batch (RECOMMANDÉ pour Windows)

#### Logger une modification:
1. Double-cliquer sur `launchers\LOG_MODIFICATION.bat`
2. Choisir la catégorie (1-8)
3. Entrer la description
4. Entrer les fichiers modifiés (optionnel)
5. Appuyer sur Entrée

#### Générer le rapport quotidien:
1. Double-cliquer sur `launchers\GENERATE_REPORT.bat`
2. Le rapport est généré automatiquement

### Option 2: Utiliser la ligne de commande (Si Python est installé)

```bash
# Logger une modification
python analysis/modification_tracker.py --action log --category FIX --message "Description" --files "fichier1.py,fichier2.py"

# Générer le rapport quotidien
python analysis/modification_tracker.py --action report

# Voir les statistiques
python analysis/modification_tracker.py --action summary

# Voir les modifications d'aujourd'hui
python analysis/modification_tracker.py --action today
```

---

## 📝 CATÉGORIES DE MODIFICATIONS

| Code | Label | Description | Exemple |
|------|-------|-------------|---------|
| **FIX** | Correction de bug | Résolution d'un problème | "Fix reward_scale=1.0 in Phase 1" |
| **FEAT** | Nouvelle fonctionnalité | Ajout de feature | "Add adaptive entropy scheduler" |
| **REFACTOR** | Refactoring | Réorganisation du code | "Refactor _calculate_reward method" |
| **DOCS** | Documentation | Mise à jour docs | "Update README with new fixes" |
| **TEST** | Tests | Ajout/modification tests | "Add unit tests for trading_env" |
| **PERF** | Performance | Optimisation | "Optimize feature calculation" |
| **CONFIG** | Configuration | Changement de config | "Update hyperparameters in train.py" |
| **DATA** | Données/Features | Modifications data | "Add new technical indicators" |

---

## 📋 EXEMPLES D'UTILISATION

### Exemple 1: Logger un Fix de Bug

**Scénario**: Vous venez de corriger le problème de reward_scale dans `trading_env.py`

**Ligne de commande**:
```bash
python analysis/modification_tracker.py --action log --category FIX --message "Fix reward_scale=1.0 pendant Phase 1 pour eviter dilution des rewards" --files "environment/trading_env.py" --details "{\"line\": 872, \"change\": \"Added condition for Phase 1\"}"
```

**Fichier Batch**:
1. Double-cliquer `LOG_MODIFICATION.bat`
2. Choisir `1` (FIX)
3. Entrer: "Fix reward_scale=1.0 pendant Phase 1 pour eviter dilution des rewards"
4. Entrer: "environment/trading_env.py"

### Exemple 2: Logger une Nouvelle Fonctionnalité

**Scénario**: Vous avez ajouté un nouveau callback dans `train.py`

**Ligne de commande**:
```bash
python analysis/modification_tracker.py --action log --category FEAT --message "Ajout du GlobalTimestepCallback pour Demonstration Learning" --files "training/train.py"
```

### Exemple 3: Logger une Mise à Jour de Documentation

**Scénario**: Vous avez mis à jour le README

**Ligne de commande**:
```bash
python analysis/modification_tracker.py --action log --category DOCS --message "Mise a jour README avec instructions de test 10K steps" --files "README.md"
```

### Exemple 4: Générer le Rapport Quotidien

**À la fin de la journée de travail**:

**Ligne de commande**:
```bash
python analysis/modification_tracker.py --action report
```

**Fichier Batch**:
1. Double-cliquer `GENERATE_REPORT.bat`

**Résultat**:
- Crée `docs/daily_reports/RAPPORT_QUOTIDIEN_20251201.md`
- Met à jour `ACTUALITE_MISE_A_JOUR.md`

### Exemple 5: Voir les Statistiques

**Ligne de commande**:
```bash
python analysis/modification_tracker.py --action summary
```

**Output**:
```
================================================================================
📊 STATISTIQUES DES MODIFICATIONS - AGENT 8
================================================================================

Total des modifications: 15

Par catégorie:
  FIX          (Correction de bug              ):   8
  FEAT         (Nouvelle fonctionnalité        ):   4
  DOCS         (Documentation                  ):   3

Fichiers les plus modifiés:
  environment/trading_env.py                          :   5 fois
  training/train.py                                   :   3 fois
  README.md                                           :   2 fois

Première modification: 2025-12-01T00:00:00
Dernière modification: 2025-12-01T18:30:00
```

---

## 🔄 WORKFLOW RECOMMANDÉ

### 1️⃣ Au Début de la Session

```
1. Consulter ACTUALITE_MISE_A_JOUR.md
   → Comprendre l'état actuel du projet

2. Lire le dernier rapport quotidien (docs/daily_reports/)
   → Voir ce qui a été fait récemment

3. Vérifier les règles dans RULES_CRITICAL.txt
   → S'assurer de respecter les contraintes
```

### 2️⃣ Pendant le Travail

```
Pour CHAQUE modification importante:

1. Faire la modification dans le code
2. Tester la modification
3. Logger la modification:
   → launchers\LOG_MODIFICATION.bat
   OU
   → python analysis/modification_tracker.py --action log ...
```

### 3️⃣ À la Fin de la Session

```
1. Générer le rapport quotidien:
   → launchers\GENERATE_REPORT.bat
   OU
   → python analysis/modification_tracker.py --action report

2. Vérifier ACTUALITE_MISE_A_JOUR.md
   → S'assurer que tout est à jour

3. Commit & Push sur GitHub:
   → launchers\PUSH_TO_GITHUB.bat
   OU
   → git add . && git commit -m "..." && git push
```

---

## ⚙️ DÉTAILS TECHNIQUES

### Format du Fichier JSON

Chaque modification est enregistrée dans `docs/MODIFICATIONS_LOG.json` avec ce format:

```json
{
  "id": 1,
  "timestamp": "2025-12-01T14:30:00",
  "date": "2025-12-01",
  "time": "14:30:00",
  "category": "FIX",
  "category_label": "Correction de bug",
  "message": "Fix reward_scale=1.0 in Phase 1",
  "files_modified": ["environment/trading_env.py"],
  "details": {
    "line": 872,
    "change": "Added condition for Phase 1"
  },
  "author": "Claude Code Agent"
}
```

### Génération des Rapports

Le rapport quotidien est généré automatiquement avec:

- **Résumé par catégorie**: Nombre de modifications par type
- **Détail complet**: Chaque modification avec timestamp, fichiers, détails
- **Statistiques globales**: Vue d'ensemble du projet
- **Instructions pour l'autre Claude**: Guide de continuation

---

## 🆘 DÉPANNAGE

### Problème: "Python was not found"

**Solution**: Utiliser les fichiers Batch (.bat) au lieu de la ligne de commande.

### Problème: Le fichier MODIFICATIONS_LOG.json n'existe pas

**Solution**: Il sera créé automatiquement à la première utilisation. Vous pouvez aussi le créer manuellement avec `[]`.

### Problème: Le rapport ne se génère pas

**Solution**:
1. Vérifier que `docs/daily_reports/` existe
2. Créer le dossier manuellement si nécessaire
3. Relancer la génération

### Problème: Je ne sais pas quelle catégorie choisir

**Guide de choix**:
- Code cassé → **FIX**
- Nouvelle feature → **FEAT**
- Réorganisation code → **REFACTOR**
- Changement README/docs → **DOCS**
- Tests unitaires → **TEST**
- Code plus rapide → **PERF**
- Hyperparams modifiés → **CONFIG**
- Ajout features/data → **DATA**

---

## 📚 RESSOURCES

### Fichiers de Référence

- [ACTUALITE_MISE_A_JOUR.md](../ACTUALITE_MISE_A_JOUR.md) - État du projet
- [RULES_CRITICAL.txt](RULES_CRITICAL.txt) - Règles à respecter
- [START_HERE.md](START_HERE.md) - Guide de démarrage
- [DIAGNOSTIC_URGENT.md](DIAGNOSTIC_URGENT.md) - Problèmes actuels

### Commandes Utiles

```bash
# Voir l'aide complète
python analysis/modification_tracker.py --help

# Logger avec tous les détails
python analysis/modification_tracker.py --action log --category FIX --message "Description" --files "file1.py,file2.py" --details '{"key": "value"}'

# Voir les modifications d'une date spécifique
python analysis/modification_tracker.py --action today --date 2025-12-01

# Générer le rapport d'une date spécifique
python analysis/modification_tracker.py --action report --date 2025-12-01
```

---

## ✅ CHECKLIST

Avant de terminer votre session, vérifiez:

- [ ] Toutes les modifications importantes sont loggées
- [ ] Le rapport quotidien est généré
- [ ] ACTUALITE_MISE_A_JOUR.md est à jour
- [ ] Les fichiers modifiés sont testés
- [ ] Commit & Push sur GitHub effectué

---

## 🤝 POUR L'AUTRE CLAUDE CODE

### Message Important

Si vous êtes un autre Claude Code qui prend la relève sur ce projet:

1. **LISEZ D'ABORD** [ACTUALITE_MISE_A_JOUR.md](../ACTUALITE_MISE_A_JOUR.md)
2. **LISEZ ENSUITE** le dernier rapport quotidien dans `docs/daily_reports/`
3. **RESPECTEZ** les règles dans [RULES_CRITICAL.txt](RULES_CRITICAL.txt)
4. **LOGGEZ** toutes vos modifications avec le système de tracking
5. **GÉNÉREZ** le rapport quotidien à la fin de votre session

### Philosophie du Système

Ce système de tracking n'est pas une bureaucratie inutile. Il permet de:

- ✅ **Traçabilité**: Comprendre pourquoi un changement a été fait
- ✅ **Collaboration**: Faciliter le travail entre plusieurs agents
- ✅ **Documentation**: Avoir un historique complet du projet
- ✅ **Débogage**: Retrouver rapidement quand un problème est apparu

**Utilisez-le systématiquement!**

---

**Dernière mise à jour**: 2025-12-01
**Version**: 1.0
**Auteur**: Claude Code Agent
