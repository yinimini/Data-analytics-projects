import pandas as pd

df = pd.read_csv("hendelser.csv")



def execute_sql_query(sql_query:str)->tuple[list[tuple], list[str]]:
  """
      Executes an SQL query and returns the fetched results.

    Args:
        sql_query (str): The SQL query to be executed.

    Returns:
        data (list): Data containing the query results.
  """
  try:
    # Establish connection
    conn = psycopg2.connect(**db_params)

    # Create a cursor object
    cur = conn.cursor()

    # Execute a SQL query
    cur.execute(sql_query)


    # Fetch data
    data = cur.fetchall()

    # Extract column names
    column_names = [desc.name for desc in cur.description]

  except (Exception, psycopg2.Error) as error:
      print(f"Error while connecting to PostgreSQL: {error}")

  finally:
      # Close the cursor and connection
      if cur:
          cur.close()
      if conn:
          conn.close()

  return data, column_names



df_filtered = df[df["year"] > 2022]

print(df_filtered)

sql_query = """
SELECT 
    veglenkesekvensid,
    relativ_posisjon,
    vegvedlikehold,
    rand_float,
    year,
    SUM(rand_float) OVER (PARTITION BY year ORDER BY relativ_posisjon) AS kumulativ_rand_float
FROM 
    hendelser
ORDER BY 
    year, relativ_posisjon;
"""
index_query = """
CREATE INDEX idx_veglenkesekvensid ON hendelser(veglenkesekvensid);


SELECT veglenkesekvensid, relativ_posisjon, vegvedlikehold, rand_float, year
FROM hendelser
WHERE veglenkesekvensid = 1507;
"""