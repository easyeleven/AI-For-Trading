import psycopg2
import pandas as pd

# Connect to the database
conn = psycopg2.connect(
    host="localhost",
    database="my_database",
    user="my_username",
    password="my_password"
)

# Load the data from Google Sheets
url = "https://docs.google.com/spreadsheets/d/1-rIkEb94tZ69FvsjXnfkVETYu6rftF-8/export?format=csv"
df = pd.read_csv(url)

# Insert each row of data into the table
cur = conn.cursor()
for index, row in df.iterrows():
    cur.execute(
        "INSERT INTO stock_data (instrument, date, open, high, low, close, volume) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (row['Instrument'], row['Date'], row['Open'], row['High'], row['Low'], row['Close'], row['Volume'])
    )
conn.commit()

# Close the database connection
cur.close()
conn.close()
