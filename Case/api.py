import psycopg2
import requests
import os 
from dotenv import load_dotenv
import json 
import pandas as pd
from sqlalchemy import create_engine

load_dotenv()
        
url = "https://nvdbapiles-v3.atlas.vegvesen.no/vegobjekter/915?inkluder=egenskaper&srid=5973&segmentering=false&trafikantgruppe=K&fylke=50&endret_etter=2018-01-01T12%3A13%3A01"

response = requests.get(url, timeout=20)

data = response.json()  

df = pd.DataFrame(data, columns=["objekter"])

# Extract and flatten JSON
rows = []
for obj in df["objekter"]:
    row = {
        "id": obj.get("id"),
        "href": obj.get("href"),
    }

    for element in obj.get("egenskaper", []):
        if "verdi" in element:
            row[element["navn"]] = element["verdi"]
        elif "innhold" in element:
            #if element["navn"] == "Liste av lokasjonsattributt":
            for innhold_item in element["innhold"]:
                row[f"{element['navn']}_id"] = innhold_item.get("id")
                row[f"{element['navn']}_navn"] = innhold_item.get("navn")
                row[f"{element['navn']}_stedfestingstype"] = innhold_item.get("stedfestingstype")
                row[f"{element['navn']}_veglenkesekvensid"] = innhold_item.get("veglenkesekvensid")
                row[f"{element['navn']}_startposisjon"] = innhold_item.get("startposisjon")
                row[f"{element['navn']}_sluttposisjon"] = innhold_item.get("sluttposisjon")
                row[f"{element['navn']}_retning"] = innhold_item.get("retning")
    
    rows.append(row)
    
df_flattened = pd.DataFrame(rows)


print(df_flattened)

username = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
database_name = os.getenv("DB_NAME")


db_url = f"postgresql://{username}:{password}@{host}:{port}/{database_name}"

engine = create_engine(db_url)
    
df_flattened.to_sql('veg_data', engine, if_exists='replace', index=False)


