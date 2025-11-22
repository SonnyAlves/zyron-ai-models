# Architecture & Contrat du Repository

> **Source de Vérité** : Ce fichier décrit l'état actuel et contractuel du repository `zyron-ai-models`.
> Toute modification structurelle doit être reflétée ici.

## 1. Arborescence Actuelle

```text
zyron-ai-models/
├── configs/
│   └── training/
│       ├── finance_7b.yaml      # Config Prod/Dev pour Finance 7B
│       └── core_13b.yaml        # Config Prod/Dev pour Core 13B
│
├── zyron/
│   ├── common/
│   │   └── data.py              # ZyronDataset (Streaming JSONL)
│   ├── finance_7b/
│   │   ├── __init__.py
│   │   └── model.py             # Classe ZyronFinance7BModel
│   └── core_13b/
│       ├── __init__.py
│       └── model.py             # Classe ZyronCore13BModel (avec Gated Cross-Attention)
│
├── scripts/
│   ├── zyron_training_orchestrator.py  # Point d'entrée unique pour le training
│   ├── zyron_nemo_smoketest.py         # Test hardware (legacy/infra)
│   └── toy_gpt_nemo.py                 # (Legacy)
│
├── data/
│   └── .gitkeep                 # Les données sont ignorées par git (montées via volume sur DGX)
│
├── models/
│   └── .gitkeep                 # Checkpoints ignorés par git
│
├── docs/
│   ├── README_ZYRON_MODELS.md
│   ├── ROADMAP.md
│   └── VISION_7B.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 2. Chemins Contractuels (API Interne)

⚠️ **ATTENTION** : Ces chemins sont hardcodés dans les scripts d'orchestration et les pipelines. Ne JAMAIS les modifier sans refactor complet.

*   **Orchestrateur** : `scripts/zyron_training_orchestrator.py`
*   **Config Finance** : `configs/training/finance_7b.yaml`
*   **Config Core** : `configs/training/core_13b.yaml`
*   **Modèle Finance** : `zyron/finance_7b/model.py`
*   **Modèle Core** : `zyron/core_13b/model.py`
*   **Checkpoints** : `models/checkpoints/` (généré automatiquement)

## 3. Workflow de Redémarrage

### A. Local (CPU / Dev Machine)

Procédure standard pour un développeur sur son laptop ou une VM sans GPU.

```bash
cd ~/Documents/zyron-ai-models

# 1. Environnement Virtuel
python3 -m venv .venv
source .venv/bin/activate

# 2. Dépendances
pip install --upgrade pip
pip install -r requirements.txt

# 3. Validation (Mode DEV sur CPU)
# Vérifie que le code compile, que les configs chargent et que la loop tourne.
python3 scripts/zyron_training_orchestrator.py --model finance_7b --mode dev
python3 scripts/zyron_training_orchestrator.py --model core_13b --mode dev
```

### B. DGX Spark (GPU / Compute Engine)

Le DGX est un moteur d'exécution piloté par Git. On ne développe pas dessus, on exécute.

```bash
# Connexion au Workbench
cd ~/nvidia-workbench/zyron-ai

# 1. Mise à jour du code
git pull origin main

# 2. Lancement du Training (Exemple)
# L'orchestrateur détectera automatiquement les GPU Blackwell/Hopper
python3 scripts/zyron_training_orchestrator.py --model finance_7b --mode prod
```

## 4. Rôles des Modèles

Les deux modèles partagent le même orchestrateur mais ont des architectures distinctes.

### 🏦 Zyron-Finance-7B
*   **Mission** : Opérations financières, conformité FR/EU, Factur-X, TVA.
*   **Architecture** : Transformer Decoder standard (style Llama/Mistral).
*   **Priorité** : Précision arithmétique et respect strict des formats JSON.

### 🧠 Zyron-Core-13B
*   **Mission** : Raisonnement complexe, structuration de données, orchestration.
*   **Architecture** : Transformer Decoder + **Gated Cross-Attention** (Visual Brain Hooks).
*   **Spécificité** : Capable de recevoir des embeddings visuels/graphiques via son mécanisme d'attention croisée.

## 5. Règles pour Contributeurs & Agents IA

1.  **Stabilité** : Ne jamais renommer les dossiers `configs/`, `scripts/` ou `zyron/` sans une raison critique.
2.  **Test Obligatoire** : Avant tout commit, lancer le test `mode dev` localement. Si ça plante sur CPU, ça plantera sur DGX.
3.  **Documentation** : Si vous modifiez l'architecture (ex: ajout d'un dossier `tools/`), vous **DEVEZ** mettre à jour ce fichier `ARCHITECTURE.md`.
4.  **Orchestrateur** : C'est le seul point d'entrée. N'ajoutez pas de scripts de training parallèles (`train_v2.py`). Améliorez l'orchestrateur existant.
