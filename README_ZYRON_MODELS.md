# Zyron AI Models - Development Repository

## 📊 Current Status

Ce repo contient :
- ✅ **Pipeline NeMo fonctionnel** sur DGX Spark (validé par smoketest)
- ✅ **Script de validation GPU** avec PyTorch 2.6 + NeMo 2.5.3
- 🔜 **Toy GPT minimal** pour tester le pipeline d'entraînement
- 📋 **Spécifications techniques** pour le futur Zyron Finance 7B

## 🚀 Ce qui fonctionne aujourd'hui

### GPU Smoke Test
```bash
python scripts/zyron_nemo_smoketest.py
# Valide : GPU NVIDIA GB10, PyTorch 2.6.0a0, NeMo 2.5.3, matmuls 8192x8192
```

### Toy GPT (en développement)
```bash
python scripts/toy_gpt_nemo.py
# Mini modèle pour valider le pipeline NeMo, PAS un vrai modèle de production
```

## 🎯 Vision : Zyron Finance 7B

**STATUS: En phase de conception**

Architecture cible envisagée :
- Modèle 7B paramètres spécialisé finance
- Fine-tuning sur données françaises/EU
- Optimisation pour DGX Spark

Voir `docs/VISION_7B.md` pour les spécifications complètes (non implémentées).

## 📂 Structure du Repo

```
scripts/         # Scripts de test et validation
models/configs/  # Futures configurations (vides pour l'instant)
notebooks/       # Expérimentations Jupyter
docs/           # Documentation et vision
data/           # Données (structure NVIDIA Workbench)
```

## 🔧 Setup Environnement

### Prérequis
- NVIDIA DGX Spark avec GPU GB10
- CUDA 12.6.3+
- Python 3.10+
- NeMo 2.5.3

### Installation
```bash
# Clone du repo
git clone git@github.com:SonnyAlves/zyron-ai-models.git
cd zyron-ai-models

# Installation des dépendances
pip install -r requirements.txt

# Test GPU
python scripts/zyron_nemo_smoketest.py
```

## 🧪 Tests Disponibles

1. **Smoke Test GPU** - Valide l'environnement DGX
   ```bash
   python scripts/zyron_nemo_smoketest.py
   ```

2. **Toy GPT** - Teste le pipeline d'entraînement (mini modèle)
   ```bash
   python scripts/toy_gpt_nemo.py
   ```

## ⚠️ Important

- Ce repo est en **développement actif**
- Zyron Finance 7B est une **vision**, pas un modèle existant
- Seuls les scripts de test GPU sont pleinement fonctionnels
- Les configurations dans `models/configs/` sont des drafts de spécification

## 📝 Roadmap

### Phase 1 - Setup & Validation ✅
- [x] Configuration DGX Spark
- [x] Validation GPU + NeMo
- [x] Structure du repo
- [ ] Toy model fonctionnel

### Phase 2 - Prototypage 🔄
- [ ] Collecte données test
- [ ] Pipeline de preprocessing
- [ ] Entraînement toy model complet

### Phase 3 - Développement 7B 📋
- [ ] Architecture finale 7B
- [ ] Datasets finance FR/EU
- [ ] Entraînement distribué
- [ ] Fine-tuning spécialisé

## 🤝 Contribution

Ce repo est privé et en développement actif. Pour toute question :
- Issues GitHub : Pour bugs et suggestions
- Contact : team@zyron.ai

## 📄 License

Propriétaire - Zyron AI © 2024