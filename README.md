# 🎨 DCGAN — Génération de visages anime

Ce projet implémente un réseau antagoniste génératif profond (**DCGAN**) avec **PyTorch** afin de générer des visages de style anime à partir d’un vecteur de bruit aléatoire.

Le modèle a été entraîné sur l’**Anime Face Dataset**, qui contient environ 60 000 images. Bien que le code d’origine ait été conçu pour une résolution de 128 × 128 pixels, les modèles pré-entraînés fournis utilisent une résolution de **64 × 64 pixels** afin de réduire le temps de calcul.

---

## 📁 Structure du dépôt

```text
.
├── DCGAN_Anime.ipynb
├── generate_images.py
├── generator.pth
├── discriminator.pth
├── requirements.txt
└── generated_images/
    ├── image_013.png
    └── image_007.png
```

* `DCGAN_Anime.ipynb` : notebook complet comprenant l’exploration des données, le prétraitement et la boucle d’entraînement.
* `generate_images.py` : script Python autonome permettant de générer de nouvelles images à partir des poids sauvegardés.
* `generator.pth` : poids pré-entraînés du générateur.
* `discriminator.pth` : poids pré-entraînés du discriminateur.
* `requirements.txt` : liste des dépendances nécessaires au projet.
* `generated_images/` : dossier contenant des exemples de visages générés.

---

## 🛠️ Architecture et stabilisation

L’entraînement d’un GAN étant souvent instable, notamment à cause du *mode collapse*, plusieurs techniques ont été intégrées pour améliorer la convergence.

### 1. Normalisation spectrale

La **Spectral Normalization** est appliquée au discriminateur. Elle contraint sa constante de Lipschitz afin d’éviter qu’il ne devienne trop confiant trop rapidement.

### 2. Taux d’apprentissage différenciés

Les taux d’apprentissage utilisés avec l’optimiseur Adam sont :

* Générateur : `0.0002`
* Discriminateur : `0.0001`

### 3. Lissage des labels

Les vraies images utilisent une cible de `0.9` au lieu de `1.0`. Cette technique empêche le discriminateur de devenir excessivement confiant.

### 4. Bruit d’instance décroissant

Un bruit est ajouté aux images réelles au début de l’entraînement. Son intensité diminue progressivement au fil des époques.

Cette technique empêche le discriminateur d’apprendre trop rapidement des différences triviales entre les vraies et les fausses images.

### 5. Suivi probabiliste

Les probabilités suivantes sont enregistrées pendant l’entraînement :

* `D(x)` : confiance du discriminateur pour les images réelles.
* `D(G(z))` : confiance du discriminateur pour les images générées.

L’objectif est de surveiller l’équilibre entre le générateur et le discriminateur, idéalement autour de `0.5`.

---

## 📦 Installation

Clonez le dépôt, puis placez-vous dans le dossier du projet :

```bash
git clone URL_DU_DEPOT
cd NOM_DU_DEPOT
```

Installez ensuite les dépendances requises :

```bash
pip install -r requirements.txt
```

---

## 💻 Utilisation

### 1. Génération d’images

Le script `generate_images.py` permet de créer de nouveaux visages anime à partir du modèle pré-entraîné `generator.pth`.

Le script est configuré pour une résolution de **64 × 64 pixels**.

Pour générer 16 images :

```bash
python generate_images.py \
  --generator_path generator.pth \
  --num_images 16 \
  --output_dir ./generated_images
```

### Génération d’une grille d’images

Pour générer des images et créer une grille récapitulative :

```bash
python generate_images.py \
  --generator_path generator.pth \
  --discriminator_path discriminator.pth \
  --grid
```

L’argument `--grid` permet de sauvegarder une seule image regroupant tous les visages générés.

Lorsque le discriminateur est fourni, le script peut également afficher le score de réalisme estimé `D(G(z))`.

---

## 🔁 Ré-entraînement du modèle

Pour lancer un nouvel entraînement, ouvrez le notebook :

```text
DCGAN_Anime.ipynb
```

Il est recommandé d’utiliser :

* Google Colab ;
* une machine équipée d’un GPU NVIDIA ;
* ou un environnement disposant de suffisamment de VRAM.

> **Attention :** le notebook contient l’architecture complète permettant de générer des images en 128 × 128 pixels. Pour revenir à une résolution de 64 × 64 pixels, retirez la dernière couche `ConvTranspose2d` du générateur et adaptez le discriminateur en conséquence.

---

## 📊 Évaluation

La qualité des images générées peut être évaluée à l’aide du **FID — Fréchet Inception Distance**.

Le notebook contient une routine utilisant `pytorch-fid` qui effectue les étapes suivantes :

1. sélection de 500 images réelles ;
2. génération de 500 images artificielles ;
3. uniformisation de la résolution des images ;
4. calcul de la distance de Fréchet entre les deux distributions.

Une valeur FID plus faible indique généralement une meilleure qualité et une plus grande diversité des images générées.

---

## 🖼️ Exemples de visages générés

<p align="center">
  <img src="generated_images/image_013.png" width="250" alt="Premier visage anime généré">
  <img src="generated_images/image_007.png" width="250" alt="Deuxième visage anime généré">
</p>

Les images ci-dessus sont chargées directement depuis le dossier `generated_images` du projet.

---

## ✍️ Auteur

**NASMANE Abdelhak**
Toulouse INP N7 — Mai 2026
