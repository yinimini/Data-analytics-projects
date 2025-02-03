import psycopg2
import requests

# Connection parameters


db_params = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': '0426',
    'host': 'localhost',
    'port': '6969'
}


try:
    # Establish connection
    conn = psycopg2.connect(**db_params)

    # Create a cursor object
    cur = conn.cursor()

    # Execute a simple query
    cur.execute("SELECT version()")
    db_version = cur.fetchall()
    print("Sucessfully connected!")
    print(f"PostgreSQL database version: {db_version[0]}")

except (Exception, psycopg2.Error) as error:
    print(f"Error while connecting to PostgreSQL: {error}")

finally:
    # Close the cursor and connection
    if cur:
        cur.close()
    if conn:
        conn.close()
        
# url = "https://nvdbapiles-v3.atlas.vegvesen.no/vegobjekter/915?inkluder=egenskaper&srid=5973&segmentering=false&trafikantgruppe=K&fylke=50&endret_etter=2018-01-01T12%3A13%3A01"
# response = requests.get(url)

# if response.status_cdoe == 200:
#     data = response.json()
# else:
#     print("")
