from pymongo import MongoClient
from datetime import datetime

url = "mongodb+srv://new:AxSTmcD1mKFH6ybo@basics.d7bdh.mongodb.net/?retryWrites=true&w=majority&appName=Basics"
#url = "mongodb+srv://naufal:naufal@intern.hd30j.mongodb.net/?retryWrites=true&w=majority&appName=intern"

# Connect to MongoDB (replace with your MongoDB URI if needed)
client = MongoClient(url)
#DB_NAME = "hivemq"
DB_NAME = "asta"
#COLLECTION_NAME = "sensor"
COLLECTION_NAME = "telemetrix"

db = client[DB_NAME]
col = db[COLLECTION_NAME]


'''num = 1
while num <= 23:
    if num < 10:
        query = {'time': '2024-11-30 0' + str(num) + ':00:00'}
        result = col.delete_one(query)
        
    else:
        query = {'time': '2024-11-30 ' + str(num) + ':00:00'}
        result = col.delete_one(query)
    num += 1

print("Done")'''

'''query = {"deviceName": "My room Temperature Sensor"}
result = col.delete_many(query)
print(result.deleted_count, "documents deleted.")'''


i = 0
cursor = col.find({})
for doc in cursor:
    i += 1

print(f"{i} total documents.")
