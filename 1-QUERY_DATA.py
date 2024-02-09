## imports the sqlite3 module in the program
# sqlite3 serve interface for interacting with SQLite databases.
import sqlite3
import pandas as pd

try:
    ## Use the connect() method of the connector class with the database name. To establish a connection to SQLite.
    conn = sqlite3.connect('retail_data.db')

    ## Use the cursor() method of a connection class to create a cursor object to execute or interact with the database SQLite command/queries from Python. 
    cursor = conn.cursor()
    print("Successfully Connected to SQLite")
    
    ## Load the data into a DataFrame
    ## Focuesd the order in UK and filter cancel order out
    sales_uk = pd.read_sql_query("SELECT * FROM retail WHERE Country = 'United Kingdom' AND Quantity > 0", conn)
    print("Retail in UK", sales_uk)

    ## Write the sales_UK dataframe to CSV file
    sales_uk.to_csv("Retail_in_UK.csv")

    ## Use cursor.clsoe() method to close the cursor
    cursor.close()

except sqlite3.Error as error:
    print("Error while connecting to sqlite", error)
finally:
    ## Use connection.clsoe() method to close the SQLite connections.
    if conn:
        conn.close()
        print("The SQLite connection is closed")
