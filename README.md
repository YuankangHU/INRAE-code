# Introduction

Le but de ce code est de pouvoir obtenir rapidement la distribution de la concentration de chlorophylle dans les lacs à n'importe quelle date.

## Télécharger le fichier

Veillez à télécharger le dossier **HydroLAKES_polys_v10_shp**, **input.txt**, **Yuankang_lac.py**. et à placer ces trois données dans le même répertoire.

## Modifier le fichier input.txt

Ouvrez le fichier input.txt

1. Dans Google Map, sélectionnez le lac, faites un clic droit n'importe où dans le lac pour obtenir les coordonnées de latitude et de longitude.

2. Collez les coordonnées dans le fichier input.txt (notez le formatage)

3. Modifier la date souhaitée

## Exécuter le fichier Yuankang_lac.py

Ouvrez le terminal de votre ordinateur

1. Assurez-vous que l'environnement python est installé (windows utiliser command **where python**)

2. Allez dans le répertoire où vous avez téléchargé le fichier

3. Entrez la commande **python Yuankang_lac.py**

## Attention

Les comptes individuels ont un nombre limité de téléchargements par mois. Vous pouvez créer votre propre compte en suivant le guide officiel (https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html#python)  

Vous obtiendrez les données **client_id** et **client_secret**, copiez-les et collez-les dans la fonction "main" de Yuankang_lac.py.  

Vous avez terminé !
