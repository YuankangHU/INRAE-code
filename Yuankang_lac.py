############################################# Introduire divers packages
import subprocess
import sys
import shutil
import os
import tarfile
import numpy as np
import matplotlib.pyplot as plt
import warnings
import rasterio
from datetime import datetime, timedelta
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image
from shapely.geometry import Point
import geopandas as gpd
from pyproj import Transformer
from skimage.draw import polygon as draw_polygon
from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import BackendApplicationClient

############################################### Définir les packages à télécharger
def install_packages():
    required_packages = [
        "numpy",
        "matplotlib",
        "rasterio",
        "pillow",       
        "shapely",
        "geopandas",
        "pyproj",
        "scikit-image",  
        "requests_oauthlib",
        "oauthlib",
        "shutil",
    ]
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Package {package} not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

########################################### Lire le fichier
def parse_input(file_path):
    coordinates = None
    start_date = None
    end_date = None
    
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('coordinate:'):
                coords = line.split('coordinate:')[1].strip()
                lat, lon = map(float, coords.split())
                coordinates = (lat, lon)
            elif line.startswith('timeRange:'):
                time_range = line.split('timeRange:')[1].strip()
                start_date, end_date = [date.strip() for date in time_range.split(',')]

    return coordinates, start_date, end_date

########################################déterminer les points de coordonnées
def check_point_in_polygon(shapefile, point):
    """
    Si le point de coordonnées d'entrée existe dans la base de données du fichier shapfile

    paramètre:
        shapefile: base de données contenant les points de coordonnées de tous les lacs
        point: point de coordonnées d'entrée
    
    retour:
        containing_polygon.iloc[0].geometry: coordonnées du polygone du lac
    """
    gdf = gpd.read_file(shapefile)

    point_geom = Point(point[1], point[0])

    containing_polygon = gdf[gdf.contains(point_geom)]

    if not containing_polygon.empty:
        return containing_polygon.iloc[0].geometry
    else:
        return None

####################################obtenir le système utm
def find_utm_zone(lat, lon):
    """
    La zone avec les coordonnées de latitude et de longitude dans le système utm

    paramètre:
        lat: latitude
        lon: longitude

    retour:
        utm_zone: par example "32632"
    """
    utm_zone = int((lon + 180) // 6) + 1
    hemisphere = '326' if lat >= 0 else '327'
    return f"{hemisphere}{utm_zone:02d}"


####################################Convertir les coordonnées en utm

def calculate_bbox(minx, miny, maxx, maxy, utm_crs):
    """
    Les coordonnées du cadre de délimitation d'entrée sont converties de WGS84 (EPSG:4326) en coordonnées UTM.

    paramètre:
        minx (float): longitude minimale
        miny (float): latitude minimale
        maxx (float): longitude maximale
        maxy (float): latitude maximale
        utm_crs (str): système utm, par example "32632": UTM zone 32N

    retour:
        bbox: bounding box [minx_utm, miny_utm, maxx_utm, maxy_utm]
    """
    transformer_to_utm = Transformer.from_crs("epsg:4326", utm_crs, always_xy=True)
    
    minx_utm, miny_utm = transformer_to_utm.transform(minx, miny)
    maxx_utm, maxy_utm = transformer_to_utm.transform(maxx, maxy)
    
    bbox = [minx_utm, miny_utm, maxx_utm, maxy_utm]
    return bbox


#################################### Ajuster la taille de la bordure
def adjust_bounding_box(bbox, resolution, utm_crs):
    # Calcul des dimensions en mètres
    width_m = bbox[2] - bbox[0]
    height_m = bbox[3] - bbox[1]
    
    # Conversion des dimensions en pixels
    width = int(width_m / resolution)
    height = int(height_m / resolution)
    
    # Détermination de la taille du pixel et du côté
    pixel_o = max(width, height)
    pixel = pixel_o + 30
    side_length_m = pixel * resolution
    
    # Calcul du centre du bbox
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    
    # Transformation des coordonnées du centre du bbox
    transformer = Transformer.from_crs(utm_crs, "epsg:4326", always_xy=True)
    center_lon, center_lat = transformer.transform(center_x, center_y)
    
    # Calcul du nouveau bbox
    new_bbox = [
        center_x - side_length_m / 2,
        center_y - side_length_m / 2,
        center_x + side_length_m / 2,
        center_y + side_length_m / 2
    ]
    
    return new_bbox, pixel



########################################## créer un masque
def create_mask(polygon, new_bbox, resolution, transformer_to_utm, coordinates, image_size):
    def transform_to_image_coords(x, y, minx_utm, maxx_utm, maxy_utm, resolution):
        x_image = (x - minx_utm) / resolution
        y_image = (maxy_utm - y) / resolution  
        x_image_flipped = image_size[1] - 1 - x_image  
        return int(x_image_flipped), int(y_image)
    
    # Création de l'image vide
    image = np.zeros(image_size, dtype=np.uint8)
    
    # Transformation des coordonnées du polygone en coordonnées d'image
    polygon_image_coords = [
        transform_to_image_coords(x, y, new_bbox[0], new_bbox[2], new_bbox[3], resolution)
        for x, y in transformer_to_utm.itransform(polygon.exterior.coords)
    ]
    
    # Extraction des coordonnées y et x pour le dessin du polygone
    poly_y, poly_x = zip(*polygon_image_coords)
    
    # Dessin du polygone sur l'image
    rr, cc = draw_polygon(poly_y, poly_x)
    image[rr, cc] = 255
    
    # Rotation de l'image
    image_rotated = np.rot90(image, k=-1)
    
    # Sauvegarde de l'image
    plt.imsave(f"{coordinates}_polygon.png", image_rotated, cmap='gray')



#plt.imshow(image_rotated, cmap='gray')
#plt.show()

############################################## authentification pour api
def authenticate_to_api(client_id, client_secret, oauth=None):
    if oauth is None:
        client = BackendApplicationClient(client_id=client_id)
        oauth = OAuth2Session(client=client)
    
    token = oauth.fetch_token(token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                              client_secret=client_secret, include_client_id=True)
    return oauth

def refresh_token_if_needed(oauth):
    token_info = oauth.token
    if token_info['expires_at'] < datetime.now().timestamp():
        print("Token expired, refreshing...")
        oauth.fetch_token(token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                          client_secret=client_secret, include_client_id=True)


################################################ Vérifier l'état de la réponse
def sentinelhub_compliance_hook(response):
    response.raise_for_status()
    return response


################################################ La fonction d'extraction du temps
def daterange(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    delta = timedelta(days=1)
    while start <= end:
        yield start
        start += delta

################################################# la fonction d'extraction du temps
def daterange(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    delta = timedelta(days=1)
    while start <= end:
        yield start
        start += delta

################################################## La position de la bouée sur la photo
def find_pixel_position(bbox, target_lat, target_lon, resolution=10):
    
    transformer_to_utm = Transformer.from_crs("epsg:4326", utm_crs, always_xy=True)

    target_x, target_y = transformer_to_utm.transform(target_lon, target_lat)

    relative_x = target_x - bbox[0]
    relative_y = target_y - bbox[1]

    pixel_x = int(relative_x / resolution)
    pixel_y = int(relative_y / resolution)

    return pixel_x, pixel_y

################################################### Zone extraite
def average_surrounding_pixels(chla, x, y, size=1):
    start_x = max(x - size, 0)
    end_x = x + size + 1
    start_y = max(y - size, 0)
    end_y = y + size + 1
    
    chla_slice = chla[start_y:end_y, start_x:end_x]
    
    if chla_slice.size == 0 or np.isnan(chla_slice).all():
        mean_chla = np.nan  
    else:
        mean_chla = np.nanmean(chla_slice)
    
    return mean_chla

################################################### Collecte des données
def process_satellite_data(oauth, evalscript, start_date, end_date, new_bbox, utm_crs, pixel, coordinates, client_id, client_secret):

    def refresh_token_if_needed(oauth):
        token_info = oauth.token
        if token_info['expires_at'] < datetime.now().timestamp():
            print("Token expired, refreshing...")
            oauth.fetch_token(token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                              client_secret=client_secret, include_client_id=True)

    for single_date in daterange(start_date, end_date):
        formatted_date = single_date.strftime("%Y-%m-%d")
        print(f"Processing data for: {formatted_date}")

        # rafraîchir le « token »
        refresh_token_if_needed(oauth)

        request = {
            "input": {
                "bounds": {
                    "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{utm_crs}"},
                    "bbox": new_bbox
                },
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {
                                "from": formatted_date + "T00:00:00Z",
                                "to": formatted_date + "T23:59:59Z",
                            },
                        },
                    }
                ],
            },
            "output": {
                "width": pixel,
                "height": pixel,
                "responses": [
                    {"identifier": "Bands", "format": {"type": "image/tiff"}},
                    {"identifier": "SCL", "format": {"type": "image/tiff"}},
                    {"identifier": "SNW", "format": {"type": "image/tiff"}},
                    {"identifier": "CLD", "format": {"type": "image/tiff"}},
                    {"identifier": "TrueColor", "format": {"type": "image/png"}},
                ],
            },
            "evalscript": evalscript,
        }

        url = "https://sh.dataspace.copernicus.eu/api/v1/process"
        
        try:
            response = oauth.post(url, json=request, headers={"Accept": "application/tar"})

            year = single_date.year
            folder_name = os.path.join(f"{coordinates}_{year}_result", formatted_date)
            os.makedirs(folder_name, exist_ok=True)

            if response.ok:
                with open(os.path.join(folder_name, 'response.tar'), 'wb') as f:
                    f.write(response.content)

                with tarfile.open(os.path.join(folder_name, 'response.tar'), 'r') as tar:
                    tar.extractall(path=folder_name)
                print(f"TIFF files extracted to {folder_name}/")
            else:
                print(f"Failed to retrieve data: {response.status_code}")

        except Exception as e:
            print(f"Error processing {formatted_date}: {e}")
            
            oauth = authenticate_to_api(client_id, client_secret)
            print("Re-authenticated and retrying...")
            response = oauth.post(url, json=request, headers={"Accept": "application/tar"})
            
            if response.ok:
                with open(os.path.join(folder_name, 'response.tar'), 'wb') as f:
                    f.write(response.content)
                with tarfile.open(os.path.join(folder_name, 'response.tar'), 'r') as tar:
                    tar.extractall(path=folder_name)
                print(f"TIFF files extracted to {folder_name}/")
            else:
                print(f"Failed to retrieve data after re-authentication: {response.status_code}")


##################################################### Nettoyage des fichiers
def check_and_delete_empty_directories(start_date, end_date,coordinates):
    for single_date in daterange(start_date, end_date):
        formatted_date = single_date.strftime("%Y-%m-%d")
        year_directory = single_date.strftime("%Y")
        current_working_directory = os.getcwd()

        directory_path = os.path.join(current_working_directory, f"{coordinates}_{single_date.year}_result", formatted_date)        
        file_path = os.path.join(directory_path, 'CLD.tif')
        file_path_SCL = os.path.join(directory_path, 'SCL.tif')

        try:
            with rasterio.open(file_path_SCL) as src:
                scl_data = src.read(1).astype('uint8')

            if os.path.exists(file_path):
                image = Image.open(file_path)
                image_array = np.array(image)

                if image_array.size == 0 or np.sum(scl_data == 6) == 0:
                    for root, dirs, files in os.walk(directory_path, topdown=False):
                        for name in files:
                            os.remove(os.path.join(root, name))
                        for name in dirs:
                            os.rmdir(os.path.join(root, name))
                    os.rmdir(directory_path)
                    print(f"Deleted empty directory: {directory_path}")
                else:
                    print(f"Directory {directory_path} contains valid data.")
            else:
                print(f"No CLD.tif file found in {directory_path}. Skipping...")

        except rasterio.errors.RasterioIOError:
            print(f"Failed to open {file_path_SCL}. File not found, skipping...")

######################################################### Fichier de bandes séparé
def split_bands(single_date, coordinates):
    """
    Divisez les bandes du fichier Bands.tif en fichiers séparés pour chaque bande.
    Si le fichier Bands.tif n'est pas trouvé, la fonction supprime le dossier single_date.
    
    Paramètres:
    single_date (str): La date du dossier en cours de traitement (format: 'YYYY-MM-DD').
    coordinates (str): Coordonnées utilisées pour construire le chemin du fichier.
    """
    
    # Définir l'année et les dossiers
    year = datetime.strptime(single_date, "%Y-%m-%d").year
    date_folder = os.path.join(f"{coordinates}_{year}_result", single_date)
    bands_directory = os.path.join(date_folder, 'bands')
    input_raster = os.path.join(date_folder, 'Bands.tif')
    output_pattern = os.path.join(bands_directory, 'band_{}.tif')
    
    # Créer le répertoire des bandes s'il n'existe pas
    os.makedirs(bands_directory, exist_ok=True)
    
    # Dictionnaire reliant les bandes à leurs noms spécifiques
    special_bands = {1: "04", 2: "05", 3: "8A", 4: "06", 5: "02", 6: "03", 7: "08", 8: "11"}
    
    # Vérifiez si Bands.tif existe
    if not os.path.exists(input_raster):
        print(f"Bands.tif not found for {single_date}. Deleting folder: {date_folder}")
        shutil.rmtree(date_folder)
        return

    try:
        # Essayez d'ouvrir le fichier Bands.tif
        with rasterio.open(input_raster) as src:
            for band in range(1, src.count + 1):
                data = src.read(band)
                
                # Déterminer le nom du fichier de sortie pour la bande actuelle
                if band in special_bands:
                    output_raster = output_pattern.format(special_bands[band])
                else:
                    output_raster = output_pattern.format(band)
                
                # Mettre à jour le profil du fichier de sortie
                profile = src.profile
                profile.update(count=1)
                
                # Écrire les données de bande dans le fichier de sortie
                with rasterio.open(output_raster, 'w', **profile) as dst:
                    dst.write(data, 1)
    
    except Exception as e:
        # Détecter toute autre erreur pouvant survenir pendant le traitement
        print(f"An error occurred while processing {single_date}: {e}")
        shutil.rmtree(date_folder)  # Nettoyer en supprimant le répertoire




######################################################## Calcul de la chlorophylle
def process_chla_map(single_date,coordinates):
    year = datetime.strptime(single_date, "%Y-%m-%d").year
    date_folder = os.path.join(f"{coordinates}_{year}_result", single_date)
    warnings.simplefilter('ignore', rasterio.errors.NotGeoreferencedWarning)
    
    with rasterio.open(os.path.join(date_folder, 'bands', 'band_04.tif')) as src:
        band_4 = src.read(1).astype(float)
    with rasterio.open(os.path.join(date_folder, 'bands', 'band_05.tif')) as src:
        band_5 = src.read(1).astype(float)
    with rasterio.open(os.path.join(date_folder, 'bands', 'band_8A.tif')) as src:
        band_8A = src.read(1).astype(float)

    with rasterio.open(f'{date_folder}/bands/band_02.tif') as src:
        band_2 = src.read(1).astype(float)
    with rasterio.open(os.path.join(date_folder, 'SCL.tif')) as src:
        scl = src.read(1).astype(float)
    with rasterio.open(os.path.join(date_folder, 'CLD.tif')) as src:
        cld = src.read(1).astype(float)

    with rasterio.open(os.path.join(date_folder, 'SNW.tif')) as src:
        snw = src.read(1).astype(float)
        
    with rasterio.open(f'{coordinates}_polygon.png') as src:
        water_mask = src.read(1).astype(bool)

    Lambda_B4 = 665
    Lambda_B5 = 705
    Lambda_B8A = 865
    bouee_latitude = 45.6622
    bouee_longitude = 2.982407

    MPH = band_5 - band_4 - ((band_8A - band_4) * (Lambda_B5 - Lambda_B4)) / (Lambda_B8A - Lambda_B4)   
    chla = 1726.50 * MPH + 18.29

    cloud_mask = np.logical_or(cld > 0, band_2 > 0.03)
    clouds_in_water = cloud_mask & water_mask

    snow_mask = snw > 0
    snow_in_water = snow_mask & water_mask
  
    water_without_clouds_snow = water_mask & ~clouds_in_water & ~ snow_in_water
    chla[~water_without_clouds_snow] = np.nan
    
    #pixel_position = find_pixel_position(new_bbox, bouee_latitude, bouee_longitude, resolution=10)
    #satellite_chla = average_surrounding_pixels(chla, pixel_position[0], pixel_position[1])

    clouds_over_water = np.where(water_mask, cld, np.nan) 
    max_CLD = np.nanmax(clouds_over_water)

    cloud_coverage_percentage = (np.sum(clouds_in_water) / np.sum(water_mask)) * 100 if np.sum(water_mask) > 0 else 0
    snow_coverage_percentage = (np.sum(snow_in_water) / np.sum(water_mask)) * 100 if np.sum(water_mask) > 0 else 0

    if cloud_coverage_percentage <= 60 and snow_coverage_percentage == 0:
        cmap = ListedColormap(["blue", "red"])
        cmap_2 = plt.get_cmap('jet')
        colors = list(cmap_2(np.linspace(0, 1, cmap_2.N)))
        colors[-1] = [1, 0, 1, 1]  # Set the top color to magenta (RGBA)
        cmap_2 = ListedColormap(colors)
        cmap_water = ListedColormap(["none", "blue"])
        cmap_cloud = ListedColormap(["none", "gray"])
        
        color_labels = np.zeros_like(chla)
        color_labels[chla > 30] = 1  
        color_labels[chla <= 30] = 0 
        color_labels[np.isnan(chla)] = np.nan 
       
        # Plotting

        fig, axs = plt.subplots(1, 2, figsize=(40, 15)) 

        # Left subplot: TrueColor image
        file_path = os.path.join(date_folder, 'TrueColor.png')
        image = Image.open(file_path)
        image = image.resize((chla.shape[1], chla.shape[0]))
        axs[0].imshow(image)
        axs[0].set_title(f'{single_date}', fontsize=30)
        axs[0].set_xticks([])
        axs[0].set_yticks([])

        # Right subplot: CHLA map
        axs[1].imshow(water_mask, cmap=cmap_water, alpha=0.1)
        axs[1].imshow(clouds_in_water, cmap=cmap_cloud, alpha= 0.9) 
        cax = axs[1].imshow(chla, cmap=cmap_2, vmin=0, vmax=60)
        max_chla = np.nanmax(chla)
        mean_chla = np.nanmean(chla)
    
        axs[1].set_title(f'Max CHLA: {max_chla:.2f} ug/L Mean CHLA: {mean_chla:.2f}', fontsize=30)

        # Colorbar
        cbar = fig.colorbar(cax, ax=axs[1], shrink=0.8, extend='max')
        cbar.set_label('Concentration (ug/L)')

        # Legend
        legend_elements = [Patch(facecolor='magenta', edgecolor='magenta', label='Chla > 60')]
        axs[1].legend(handles=legend_elements, loc='upper right', fontsize='large', bbox_to_anchor=(0.98, 0.98))

        axs[1].set_xticks([])
        axs[1].set_yticks([])
        
        current_working_directory = os.getcwd()
        new_directory = os.path.join(current_working_directory, f"{coordinates}_{year}_movie")
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)

        file_path = os.path.join(new_directory, f"{single_date}.tif")
        plt.tight_layout()  
        plt.savefig(file_path, bbox_inches='tight', pad_inches=1)  
        #plt.show()
        plt.close(fig)


############################################### Main Function
def main():
    # Installer les packages requis
    install_packages()

    # Chemins d'accès au fichier d'entrée et au fichier de formes
    input_file_path = 'input.txt'
    shapefile = "HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10.shp"

    # Analyser le fichier d'entrée
    coordinates, start_date, end_date = parse_input(input_file_path)

    # Générer une boîte englobante et trouver une zone UTM
    if coordinates:
        polygon = check_point_in_polygon(shapefile, coordinates)
        if polygon:
            minx, miny, maxx, maxy = polygon.bounds
            print(f"Coordinates: {coordinates}")
            print(f"Time Range: {start_date} to {end_date}")
            print(f"Polygon Bounds: minx={minx}, miny={miny}, maxx={maxx}, maxy={maxy}")

            # Rechercher le CRS UTM et ajuster le cadre de délimitation
            utm_crs = find_utm_zone(coordinates[0], coordinates[1])
            transformer_to_utm = Transformer.from_crs("epsg:4326", utm_crs, always_xy=True)
            bbox = calculate_bbox(minx, miny, maxx, maxy, utm_crs)
            resolution = 10
            new_bbox, pixel = adjust_bounding_box(bbox, resolution, utm_crs)

            # Créer un masque
            image_size = (pixel, pixel)
            create_mask(polygon, new_bbox, resolution, transformer_to_utm, coordinates, image_size)
            
            # Authentifier auprès de l'API
            client_id = 'sh-ba4df285-f046-4cee-9f15-f4c1abc9f1c1'
            client_secret = 'HlFXAyaPhbY23NS1Jn3bIjxBpLM3AGcD'
            oauth = authenticate_to_api(client_id, client_secret)
            oauth.register_compliance_hook("access_token_response", sentinelhub_compliance_hook)

            # Traiter les données satellite
            evalscript = """
            //VERSION=3
            function setup() {
              return {
                input: [
                  {bands: ["B04", "B05", "B8A", "B06", "SCL", "CLD", "SNW", "B02", "B03", "B08", "B11"]},
                ],
                output: [
                  {id: "Bands", bands: 8, sampleType: SampleType.FLOAT32},
                  {id: "SCL", bands: 1, sampleType: SampleType.UINT8},
                  {id: "CLD", bands: 1, sampleType: SampleType.UINT8},
                  {id: "SNW", bands: 1, sampleType: SampleType.UINT8},
                  {id: "TrueColor", bands: 3, sampleType: SampleType.UINT8},
                ],
              }
            }

            function evaluatePixel(sample) {
              return {
                Bands: [sample.B04, sample.B05, sample.B8A, sample.B06, sample.B02, sample.B03, sample.B08, sample.B11],
                SCL: [sample.SCL],
                CLD: [sample.CLD],
                SNW: [sample.SNW],
                TrueColor: [2.5 * sample.B04 * 255, 2.5 * sample.B03 * 255, 2.5 * sample.B02 * 255],
              }
            }
            """
            process_satellite_data(oauth, evalscript, start_date, end_date, new_bbox, utm_crs, pixel, coordinates, client_id, client_secret)
            check_and_delete_empty_directories(start_date, end_date,coordinates)

            start_year = datetime.strptime(start_date, "%Y-%m-%d").year
            end_year = datetime.strptime(end_date, "%Y-%m-%d").year
            for year in range(start_year, end_year + 1):
                directories = sorted(os.listdir(f'{coordinates}_{year}_result'))
                for date in directories:
                    split_bands(date,coordinates)
                    process_chla_map(date,coordinates)


                    

            
        else:
            print("The point is not within any lake polygon.")
    else:
        print("No valid coordinates found in the input file.")


# Exécuter la fonction principale
if __name__ == "__main__":
    main()