import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="train",
  password="train"
)

print(mydb)