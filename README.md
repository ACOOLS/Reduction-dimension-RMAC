# TFE
Le but de mon TFE est de pouvoir créer un moteur de recherche pour les classes et les sous-classes (dans mon cas j'ai choisi les voitures).

- D'abord tester les architectures les plus connues avec les poids de **ImageNet** sur la base d'images **GHIM-20**
- Entrainer la base d'images **GHIM-20** avec ces architectures
- Combiner les meilleures architectures
- Appliquer l'algorithme **RMAC**, **MS-RMAC**, **RMAC+**
- Entrainer notre base d'images construites avec les architectures retenues.
- Application web de notre moteur de recherche
# Premier pas
```sh
git clone https://github.com/ACOOLS/Memoire.git
sudo apt update
sudo apt install libgl1-mesa-glx
```
# REQUIREMENT
## Indexation et recherche
```sh
pip install -r requirements/requirement_imm.txt
```
## RMAC
### python env (pip)
```sh
pip install -r requirements/requirement_rmac.txt
```
### Ananconda env
```sh
 conda create -n rmac --file requirements/anaconda_rmac.txt
 conda activate rmac
```
### Docker
#### Build docker image
```sh
 cd docker
 docker build -t acools/tfe .
 ```
# Implémentation
## Indexation && recherche
### GHIM-20
Nous vous recommandons de passer d'abord par cette étape, nous avons automatisé le code de sorte qu'il télécharge la base d'images puis prépape la base pour l'entrainnement, l'indexation et la recherche.
```sh
cd normal
python Train.py
```
### Base d'images de voitures
Pour télécharger la base contactez moi afin de rendre le projet public
```sh
bash download_cars_data_set.sh
cd normal
python train_cars.py
```
## Recherche Multiple

 ```sh
cd normal
 python Recherche_Multi_arch.py
```
## Fusionner les modèles
 ```sh
cd Merge_Models
 python merge_models.py
```
## Compression
### RMAC 
#### Indexation
```sh
 cd RMAC
python RMAC_indexation

```
#### Recherche par architecture
```sh
 cd RMAC
 python  RMAC_recherche.py
```
### MS-RMAC
```sh
cd MS-RMAC
python MSRMAC_Indexation.py 
```
#### Recherche par architecture
```sh
 cd MS-RMAC
 python  Recherche.py
```
### RMAC+
```sh
cd RMAC+
python rmac_plus.py
```
### Moteur de recherche 
```sh
cd Moteur_Recherche
python moteur_recherche.py --image lien_image --model modèle choisi
```
## License

Aurélie Cools
