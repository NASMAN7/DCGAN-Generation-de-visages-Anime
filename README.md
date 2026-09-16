🎨 DCGAN — Génération de visages anime

Ce projet implémente un réseau antagoniste génératif profond (DCGAN) avec PyTorch afin de générer des visages de style anime à partir d’un vecteur de bruit aléatoire.

Le modèle a été entraîné sur l’Anime Face Dataset, qui contient environ 60 000 images. Bien que le code d’origine ait été conçu pour une résolution de 128 × 128 pixels, les modèles pré-entraînés fournis utilisent une résolution de 64 × 64 pixels afin de réduire le temps de calcul.

📁 Structure du dépôt

.
├── DCGAN_Anime.ipynb
├── generate_images.py
├── generator.pth
├── discriminator.pth
├── requirements.txt
└── generated_images/

DCGAN_Anime.ipynb : notebook complet comprenant l’exploration des données, le prétraitement et la boucle d’entraînement.

generate_images.py : script Python autonome permettant de générer de nouvelles images à partir des poids sauvegardés.

generator.pth et discriminator.pth : poids pré-entraînés des modèles en résolution 64 × 64.

requirements.txt : liste des dépendances nécessaires au projet.

generated_images/ : dossier contenant des exemples de visages générés.

🛠️ Architecture et stabilisation

L’entraînement d’un GAN étant souvent instable — notamment à cause du mode collapse — plusieurs techniques ont été intégrées afin d’améliorer la convergence :

Normalisation spectrale — discriminateur : contraint la constante de Lipschitz afin d’éviter que le discriminateur ne devienne trop confiant trop rapidement.

Taux d’apprentissage différenciés : 0.0002 pour le générateur et 0.0001 pour le discriminateur, avec l’optimiseur Adam.

Lissage des labels : utilise une cible de 0.9 au lieu de 1.0 pour les images réelles.

Bruit d’instance décroissant : ajoute du bruit aux images réelles au début de l’entraînement, puis le réduit progressivement au fil des époques. Cela empêche le discriminateur d’apprendre des raccourcis triviaux.

Suivi probabiliste : journalise D(x), la confiance accordée aux images réelles, et D(G(z)), la confiance accordée aux images générées, afin de surveiller leur équilibre théorique autour de 0.5.

📦 Installation

Clonez le dépôt, placez-vous dans son dossier, puis installez les dépendances :

pip install -r requirements.txt

💻 Utilisation

1. Générer des images — inférence

Utilisez generate_images.py pour créer de nouveaux visages à partir du modèle pré-entraîné generator.pth. Le script est configuré pour l’architecture 64 × 64.

Générer 16 images :

python generate_images.py \
  --generator_path generator.pth \
  --num_images 16 \
  --output_dir ./generated_images

Générer des images et afficher le score de réalisme estimé D(G(z)) :

python generate_images.py \
  --generator_path generator.pth \
  --discriminator_path discriminator.pth \
  --grid

L’argument --grid permet de sauvegarder une grille récapitulative regroupant toutes les images générées.

2. Ré-entraîner le modèle

Pour lancer un nouvel entraînement, ouvrez DCGAN_Anime.ipynb, idéalement dans Google Colab ou sur une machine équipée d’un GPU.

[!WARNING]
Le notebook contient l’architecture complète pour générer des images en 128 × 128. Pour l’exécuter tel quel, prévoyez suffisamment de temps de calcul et de VRAM. Pour revenir à une résolution de 64 × 64, retirez la dernière couche ConvTranspose2d du générateur et adaptez le discriminateur en conséquence.

📊 Évaluation

La qualité des images générées peut être évaluée avec le FID — Fréchet Inception Distance. Le notebook contient une routine utilisant pytorch-fid qui :

isole un échantillon de 500 images réelles ;

génère 500 images artificielles ;

uniformise les résolutions ;

calcule la distance de Fréchet entre les deux distributions.

Exemples de visages générés

![Visage anime généré](generated_images/image_013.png)

![Deuxième visage généré](generated_images/image_007.png)





✍️ Auteur

NASMANE Abdelhak
Toulouse INP N7 — Mai 2026

