# Introduction

Le but de ce code est de pouvoir obtenir rapidement la distribution de la concentration de chlorophylle dans les lacs à n'importe quelle date.

## Télécharger le fichier

1. lien pour **HydroLAKES_polys_v10_shp**: https://drive.google.com/drive/folders/1CskSfQ6mfED9BvveSRb592PIxQrtJdhd?usp=drive_link
2. Veillez à télécharger les fichiers **HydroLAKES_polys_v10_shp**, **input.txt**, **credentials.txt**,**Yuankang_lac.py**. et à placer ces quatre données dans le même répertoire.

## Créez un compte sur le site officiel du satellite Copernicus

1. Site officiel:https://dataspace.copernicus.eu/
2. Obtenez le token pour vous connecter au serveur: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html#python
3. Selon le token obtenu. Copiez et collez le client_id et le client_secret obtenus dans le fichier **credentials.txt**
4. Attention à ne pas avoir d'espaces de part et d'autre de la virgule

## Modifier le fichier input.txt

Ouvrez le fichier input.txt

1. Dans Google Map, sélectionnez le lac, faites un clic droit n'importe où dans le lac pour obtenir les coordonnées de latitude et de longitude.

2. Collez les coordonnées dans le fichier input.txt

3. Remplissez les quatre valeurs: **latitude**, **longitude**, **start_date**, **end_date** dans l'ordre

## Exécuter le fichier Yuankang_lac.py

Ouvrez le terminal de votre ordinateur

1. Assurez-vous que l'environnement python est installé (windows utiliser command **where python**)

2. Allez dans le répertoire où vous avez téléchargé le fichier

3. Entrez la commande **python Yuankang_lac.py**

## Attention

Les comptes individuels ont un nombre limité de téléchargements par mois. Vous pouvez vous inscrire avec différents comptes de email et vous pouvez vérifier l'utilisation sur votre page personnelle sur le site officiel.


Vous avez terminé !
