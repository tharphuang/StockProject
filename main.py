from infra.mongo import get_mongodb

client, db = get_mongodb("test")
collection = db["user_info"]

myDict = {"user": "tharp", "password": "123", "is_delete": False}
collection.insert_one(myDict)

client.close()
