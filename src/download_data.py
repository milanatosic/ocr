# src/download_data.py
from roboflow import Roboflow

# Inicijalizacija sa tvojim tajnim ključem
rf = Roboflow(api_key="f53u4350cMewtGprlCIA")

# Učitavanje projekta i verzije u kojoj su sve slike u Train skupu
project = rf.project("inteligentni-sistemi-qpjez")
dataset = project.version(4).download("yolov8")

print("Dataset je uspešno preuzet u koren projekta")