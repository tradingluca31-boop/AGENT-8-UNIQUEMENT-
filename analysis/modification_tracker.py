"""
AGENT 8 - Modification Tracker & Daily Report Generator
================================================================================
Script pour tracker automatiquement toutes les modifications du projet
et générer un rapport quotidien pour faciliter la collaboration entre Claude Code agents

Version: 1.0
Date création: 2025-12-01
Auteur: Claude Code Agent

Usage:
    python analysis/modification_tracker.py --action log --message "Description de la modification"
    python analysis/modification_tracker.py --action report
    python analysis/modification_tracker.py --action summary
================================================================================
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import subprocess

# ================================================================================
# CONFIGURATION
# ================================================================================

# Dossier racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Fichier de log des modifications
MODIFICATIONS_LOG_FILE = PROJECT_ROOT / "docs" / "MODIFICATIONS_LOG.json"

# Fichier de rapport quotidien
DAILY_REPORT_DIR = PROJECT_ROOT / "docs" / "daily_reports"
DAILY_REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Catégories de modifications
CATEGORIES = {
    "FIX": "Correction de bug",
    "FEAT": "Nouvelle fonctionnalité",
    "REFACTOR": "Refactoring de code",
    "DOCS": "Documentation",
    "TEST": "Tests",
    "PERF": "Optimisation de performance",
    "CONFIG": "Configuration",
    "DATA": "Données/Features"
}

# ================================================================================
# CLASSES
# ================================================================================

class ModificationTracker:
    """Classe pour tracker et gérer les modifications du projet"""

    def __init__(self):
        self.log_file = MODIFICATIONS_LOG_FILE
        self.modifications = self._load_modifications()

    def _load_modifications(self) -> List[Dict[str, Any]]:
        """Charge les modifications depuis le fichier JSON"""
        if self.log_file.exists():
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def _save_modifications(self):
        """Sauvegarde les modifications dans le fichier JSON"""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.modifications, f, indent=2, ensure_ascii=False)

    def log_modification(self, category: str, message: str, files_modified: List[str] = None,
                         details: Dict[str, Any] = None):
        """
        Enregistre une nouvelle modification

        Args:
            category: Catégorie de modification (FIX, FEAT, etc.)
            message: Description de la modification
            files_modified: Liste des fichiers modifiés
            details: Détails additionnels (paramètres changés, lignes modifiées, etc.)
        """
        modification = {
            "id": len(self.modifications) + 1,
            "timestamp": datetime.now().isoformat(),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "category": category.upper(),
            "category_label": CATEGORIES.get(category.upper(), "Autre"),
            "message": message,
            "files_modified": files_modified or [],
            "details": details or {},
            "author": "Claude Code Agent"
        }

        self.modifications.append(modification)
        self._save_modifications()

        print(f"✅ Modification #{modification['id']} enregistrée:")
        print(f"   Catégorie: {modification['category']} - {modification['category_label']}")
        print(f"   Message: {message}")
        if files_modified:
            print(f"   Fichiers: {', '.join(files_modified)}")
        print()

    def get_modifications_by_date(self, date: str = None) -> List[Dict[str, Any]]:
        """Récupère les modifications pour une date donnée (format YYYY-MM-DD)"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        return [mod for mod in self.modifications if mod.get("date") == date]

    def get_recent_modifications(self, n: int = 10) -> List[Dict[str, Any]]:
        """Récupère les N dernières modifications"""
        return self.modifications[-n:] if len(self.modifications) >= n else self.modifications

    def get_statistics(self) -> Dict[str, Any]:
        """Calcule des statistiques sur les modifications"""
        total = len(self.modifications)

        # Comptage par catégorie
        by_category = {}
        for mod in self.modifications:
            cat = mod.get("category", "UNKNOWN")
            by_category[cat] = by_category.get(cat, 0) + 1

        # Comptage par date
        by_date = {}
        for mod in self.modifications:
            date = mod.get("date", "unknown")
            by_date[date] = by_date.get(date, 0) + 1

        # Fichiers les plus modifiés
        file_counts = {}
        for mod in self.modifications:
            for file in mod.get("files_modified", []):
                file_counts[file] = file_counts.get(file, 0) + 1

        top_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_modifications": total,
            "by_category": by_category,
            "by_date": by_date,
            "top_modified_files": dict(top_files),
            "first_modification": self.modifications[0].get("timestamp") if self.modifications else None,
            "last_modification": self.modifications[-1].get("timestamp") if self.modifications else None
        }

# ================================================================================
# RAPPORT QUOTIDIEN
# ================================================================================

class DailyReportGenerator:
    """Génère un rapport quotidien des modifications pour faciliter la collaboration"""

    def __init__(self, tracker: ModificationTracker):
        self.tracker = tracker

    def generate_report(self, date: str = None) -> str:
        """
        Génère un rapport détaillé pour une date donnée

        Args:
            date: Date au format YYYY-MM-DD (défaut: aujourd'hui)

        Returns:
            Chemin du fichier de rapport généré
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        modifications = self.tracker.get_modifications_by_date(date)

        # Générer le contenu du rapport
        report_content = self._generate_report_content(date, modifications)

        # Sauvegarder le rapport
        report_filename = f"RAPPORT_QUOTIDIEN_{date.replace('-', '')}.md"
        report_path = DAILY_REPORT_DIR / report_filename

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        # Mettre à jour le fichier ACTUALITE_MISE_A_JOUR.md
        self._update_main_log(date, modifications)

        return str(report_path)

    def _generate_report_content(self, date: str, modifications: List[Dict[str, Any]]) -> str:
        """Génère le contenu Markdown du rapport quotidien"""

        content = f"""# 📋 RAPPORT QUOTIDIEN - AGENT 8

**Date**: {date}
**Généré le**: {datetime.now().strftime("%Y-%m-%d à %H:%M:%S")}
**Nombre de modifications**: {len(modifications)}

---

## 📌 RÉSUMÉ DE LA JOURNÉE

"""

        if not modifications:
            content += """**Aucune modification enregistrée aujourd'hui.**

C'est peut-être un jour de repos, d'analyse ou de tests sans modifications de code.

"""
        else:
            # Grouper par catégorie
            by_category = {}
            for mod in modifications:
                cat = mod.get("category", "UNKNOWN")
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(mod)

            # Résumé par catégorie
            content += "### Par catégorie:\n\n"
            for cat, mods in sorted(by_category.items()):
                cat_label = CATEGORIES.get(cat, "Autre")
                content += f"- **{cat}** ({cat_label}): {len(mods)} modification(s)\n"

            content += "\n---\n\n"

            # Détail des modifications
            content += "## 🔧 DÉTAIL DES MODIFICATIONS\n\n"

            for i, mod in enumerate(modifications, 1):
                cat = mod.get("category", "UNKNOWN")
                cat_label = CATEGORIES.get(cat, "Autre")
                time = mod.get("time", "00:00:00")
                message = mod.get("message", "Pas de description")
                files = mod.get("files_modified", [])
                details = mod.get("details", {})

                content += f"### Modification #{i} - {time}\n\n"
                content += f"**Catégorie**: {cat} - {cat_label}\n\n"
                content += f"**Description**: {message}\n\n"

                if files:
                    content += "**Fichiers modifiés**:\n"
                    for file in files:
                        content += f"- [{file}]({file})\n"
                    content += "\n"

                if details:
                    content += "**Détails**:\n```json\n"
                    content += json.dumps(details, indent=2, ensure_ascii=False)
                    content += "\n```\n\n"

                content += "---\n\n"

        # Statistiques globales
        stats = self.tracker.get_statistics()

        content += """## 📊 STATISTIQUES GLOBALES DU PROJET

"""
        content += f"**Total des modifications**: {stats['total_modifications']}\n\n"

        content += "**Par catégorie (total projet)**:\n"
        for cat, count in sorted(stats['by_category'].items(), key=lambda x: x[1], reverse=True):
            cat_label = CATEGORIES.get(cat, "Autre")
            content += f"- {cat} ({cat_label}): {count}\n"

        content += "\n**Fichiers les plus modifiés**:\n"
        for file, count in list(stats['top_modified_files'].items())[:5]:
            content += f"- {file}: {count} fois\n"

        content += f"""

---

## 🎯 POUR L'AUTRE CLAUDE CODE

### Comment utiliser ce rapport:

1. **Lire ce rapport** pour comprendre ce qui a été fait aujourd'hui
2. **Vérifier les fichiers modifiés** pour comprendre les changements
3. **Consulter les détails** pour voir les paramètres exacts modifiés
4. **Continuer le travail** en respectant les règles dans [RULES_CRITICAL.txt](../RULES_CRITICAL.txt)

### Règles à respecter:

- ❌ **NE JAMAIS** créer de nouvelles versions (V2.8, V3.0, etc.)
- ✅ **TOUJOURS** modifier directement les fichiers existants
- ✅ **TOUJOURS** logger les modifications avec `modification_tracker.py`
- ✅ **TOUJOURS** tester après chaque modification
- ✅ **TOUJOURS** pusher sur GitHub après modifications importantes

### Commandes utiles:

```bash
# Logger une modification
python analysis/modification_tracker.py --action log --category FIX --message "Description"

# Générer le rapport quotidien
python analysis/modification_tracker.py --action report

# Voir le résumé des modifications
python analysis/modification_tracker.py --action summary
```

---

**Généré automatiquement par**: [modification_tracker.py](../analysis/modification_tracker.py)
**Dernière mise à jour**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

        return content

    def _update_main_log(self, date: str, modifications: List[Dict[str, Any]]):
        """Met à jour le fichier principal ACTUALITE_MISE_A_JOUR.md"""

        main_log_path = PROJECT_ROOT / "ACTUALITE_MISE_A_JOUR.md"

        if not main_log_path.exists():
            print(f"⚠️  Fichier {main_log_path} introuvable, création...")
            return

        # Lire le contenu actuel
        with open(main_log_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Préparer la section de mise à jour
        update_section = f"\n\n---\n\n## 🔄 DERNIÈRE MISE À JOUR: {date}\n\n"
        update_section += f"**Nombre de modifications**: {len(modifications)}\n\n"

        if modifications:
            update_section += "**Modifications récentes**:\n"
            for mod in modifications[-5:]:  # Les 5 dernières
                cat = mod.get("category", "UNKNOWN")
                message = mod.get("message", "Pas de description")
                time = mod.get("time", "00:00:00")
                update_section += f"- `{time}` **[{cat}]** {message}\n"

            update_section += f"\n**Rapport complet**: [RAPPORT_QUOTIDIEN_{date.replace('-', '')}.md](docs/daily_reports/RAPPORT_QUOTIDIEN_{date.replace('-', '')}.md)\n"

        # Remplacer la section "Dernière mise à jour" si elle existe
        if "## 🔄 DERNIÈRE MISE À JOUR:" in content:
            # Trouver le début de la section
            start_idx = content.find("## 🔄 DERNIÈRE MISE À JOUR:")
            # Trouver la prochaine section (ou la fin)
            next_section_idx = content.find("\n## ", start_idx + 10)
            if next_section_idx == -1:
                next_section_idx = len(content)

            # Remplacer
            content = content[:start_idx] + update_section + content[next_section_idx:]
        else:
            # Ajouter à la fin
            content += update_section

        # Mettre à jour la date en haut du fichier
        content = content.replace(
            "**Dernière mise à jour**:",
            f"**Dernière mise à jour**: {date} - "
        )

        # Sauvegarder
        with open(main_log_path, 'w', encoding='utf-8') as f:
            f.write(content)

# ================================================================================
# CLI
# ================================================================================

def main():
    """Point d'entrée principal du script"""

    parser = argparse.ArgumentParser(
        description="AGENT 8 - Modification Tracker & Daily Report Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

  # Logger une correction de bug
  python modification_tracker.py --action log --category FIX --message "Correction du reward_scale en Phase 1" --files "environment/trading_env.py"

  # Logger une nouvelle feature
  python modification_tracker.py --action log --category FEAT --message "Ajout de l'adaptive entropy" --files "training/train.py"

  # Générer le rapport quotidien
  python modification_tracker.py --action report

  # Voir le résumé des modifications
  python modification_tracker.py --action summary

  # Voir les modifications d'aujourd'hui
  python modification_tracker.py --action today
        """
    )

    parser.add_argument(
        '--action',
        choices=['log', 'report', 'summary', 'today'],
        required=True,
        help="Action à effectuer"
    )

    parser.add_argument(
        '--category',
        choices=list(CATEGORIES.keys()),
        help="Catégorie de modification (requis pour --action log)"
    )

    parser.add_argument(
        '--message',
        type=str,
        help="Description de la modification (requis pour --action log)"
    )

    parser.add_argument(
        '--files',
        type=str,
        help="Fichiers modifiés, séparés par des virgules (optionnel)"
    )

    parser.add_argument(
        '--details',
        type=str,
        help="Détails additionnels au format JSON (optionnel)"
    )

    parser.add_argument(
        '--date',
        type=str,
        help="Date au format YYYY-MM-DD (défaut: aujourd'hui)"
    )

    args = parser.parse_args()

    # Initialiser le tracker
    tracker = ModificationTracker()

    # Exécuter l'action demandée
    if args.action == 'log':
        # Validation
        if not args.category or not args.message:
            parser.error("--category et --message sont requis pour --action log")

        # Parser les fichiers
        files_modified = None
        if args.files:
            files_modified = [f.strip() for f in args.files.split(',')]

        # Parser les détails
        details = None
        if args.details:
            try:
                details = json.loads(args.details)
            except json.JSONDecodeError:
                print("⚠️  Erreur: --details doit être un JSON valide")
                sys.exit(1)

        # Logger la modification
        tracker.log_modification(
            category=args.category,
            message=args.message,
            files_modified=files_modified,
            details=details
        )

        print("✅ Modification enregistrée avec succès!")
        print("\n💡 N'oubliez pas de générer le rapport quotidien avec:")
        print("   python modification_tracker.py --action report")

    elif args.action == 'report':
        # Générer le rapport quotidien
        generator = DailyReportGenerator(tracker)
        report_path = generator.generate_report(date=args.date)

        print(f"✅ Rapport quotidien généré:")
        print(f"   {report_path}")
        print("\n📄 Le fichier ACTUALITE_MISE_A_JOUR.md a également été mis à jour.")

    elif args.action == 'summary':
        # Afficher les statistiques
        stats = tracker.get_statistics()

        print("="*80)
        print("📊 STATISTIQUES DES MODIFICATIONS - AGENT 8")
        print("="*80)
        print()
        print(f"Total des modifications: {stats['total_modifications']}")
        print()
        print("Par catégorie:")
        for cat, count in sorted(stats['by_category'].items(), key=lambda x: x[1], reverse=True):
            cat_label = CATEGORIES.get(cat, "Autre")
            print(f"  {cat:12s} ({cat_label:30s}): {count:3d}")
        print()
        print("Fichiers les plus modifiés:")
        for file, count in list(stats['top_modified_files'].items())[:10]:
            print(f"  {file:50s}: {count:3d} fois")
        print()
        print(f"Première modification: {stats.get('first_modification', 'N/A')}")
        print(f"Dernière modification: {stats.get('last_modification', 'N/A')}")
        print()
        print("="*80)

    elif args.action == 'today':
        # Afficher les modifications d'aujourd'hui
        date = args.date or datetime.now().strftime("%Y-%m-%d")
        modifications = tracker.get_modifications_by_date(date)

        print("="*80)
        print(f"📋 MODIFICATIONS DU {date}")
        print("="*80)
        print()

        if not modifications:
            print("Aucune modification enregistrée pour cette date.")
        else:
            print(f"Nombre de modifications: {len(modifications)}")
            print()

            for i, mod in enumerate(modifications, 1):
                cat = mod.get("category", "UNKNOWN")
                cat_label = CATEGORIES.get(cat, "Autre")
                time = mod.get("time", "00:00:00")
                message = mod.get("message", "Pas de description")
                files = mod.get("files_modified", [])

                print(f"#{i} - {time}")
                print(f"   Catégorie: {cat} ({cat_label})")
                print(f"   Message: {message}")
                if files:
                    print(f"   Fichiers: {', '.join(files)}")
                print()

        print("="*80)

if __name__ == "__main__":
    main()
