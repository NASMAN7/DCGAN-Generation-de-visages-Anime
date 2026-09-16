# 🎨 DCGAN — Génération de Visages Anime

Ce projet implémente un réseau de neurones antagoniste génératif profond (**DCGAN**) avec PyTorch pour générer des visages de style anime à partir d'un vecteur de bruit aléatoire. 

Le modèle a été entraîné sur l'**Anime Face Dataset** (~60 000 images). Bien que le code d'origine ait été conçu pour une résolution de 128x128, **les modèles fournis ont été entraînés en 64x64** afin d'optimiser les temps de calcul.

---

## 📁 Structure du dépôt

Comme illustré dans le dépôt, voici les fichiers principaux :

*   `DCGAN_Anime.ipynb` : Notebook complet contenant la phase d'exploration, le prétraitement et la boucle d'entraînement.
*   `generate_images.py` : Script Python autonome pour générer de nouvelles images à partir des poids sauvegardés.
*   `generator.pth` & `discriminator.pth` : Poids pré-entraînés de nos modèles (résolution 64x64).
*   `requirements.txt` : Liste des dépendances nécessaires au projet.
*   `generated_images/` : Dossier contenant des exemples de visages générés.

---

## 🛠️ Architecture & Stabilisation

L'entraînement d'un GAN étant souvent instable (ex: *mode collapse*), plusieurs techniques de pointe ont été intégrées pour stabiliser la convergence :

1.  **Spectral Normalization** (Discriminateur) : Contraint la constante de Lipschitz pour éviter que le discriminateur ne devienne trop confiant trop rapidement.
2.  **Learning Rates Différenciés** : (Générateur: `0.0002`, Discriminateur: `0.0001` avec optimiseur Adam).
3.  **Label Smoothing** : Utilisation de labels cibles à `0.9` (au lieu de `1.0`) pour les vraies images.
4.  **Bruit d'instance décroissant** : Ajout d'un bruit sur les images réelles en début d'entraînement, qui diminue avec les epochs, empêchant le discriminateur d'apprendre des raccourcis triviaux.
5.  **Suivi probabiliste** : Logging de `D(x)` (confiance sur les images réelles) et `D(G(z))` (confiance sur les images générées) pour viser l'équilibre théorique de `0.5`.

---

## 🚀 Installation

Clonez ce dépôt sur votre machine locale :

```bash
git clone [https://github.com/NASMAN7/DCGAN-Generation-de-visages-Anime.git](https://github.com/NASMAN7/DCGAN-Generation-de-visages-Anime.git)
cd DCGAN-Generation-de-visages-Anime
