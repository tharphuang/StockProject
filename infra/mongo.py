from pymongo import MongoClient


def get_mongodb(db_name):
    """
    :param db_name: 数据库名
    :return: 数据库实例
    """
    MONGO_USER_NAME = "root"
    MONGO_PSW = "password"
    MONGO_HOST = "localhost"
    MONGO_PORT = 27017
    if not MONGO_USER_NAME or not MONGO_PSW:  # 不需要认证
        if (not MONGO_HOST) and (not MONGO_PORT):
            client = MongoClient()
        else:
            client = MongoClient(MONGO_HOST, int(MONGO_PORT))
        db = client[db_name]
        return client, db
    else:
        client = MongoClient(MONGO_HOST, MONGO_PORT)
        db = client.admin
        db.authenticate(MONGO_USER_NAME, MONGO_PSW, mechanism='SCRAM-SHA-1')
        db = client[db_name]
        return client, db
