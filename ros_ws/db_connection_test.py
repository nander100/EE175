import os
import sys
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError

def test_mongo_connection():
    print("[1/1] MONGODB CONNECTION TEST")
    
    try:
        connection_string = os.environ.get('DB_URI')
        if not connection_string:
            print("----- ERROR: DB_URI environment variable is not set.")
            print("----- Make sure you are running this script inside the Docker container.")
            sys.exit(1)
        print(f"Attempting to connect with URI: {connection_string[:20]}...")
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        
        print("\n----- SUCCESS: Connection to MongoDB is successful!")
        info = client.server_info()
        print(f"----- Server Version: {info['version']}")
        
    except ConnectionFailure as e:
        print("\n----- FAILED: Could not connect to MongoDB.")
        print("----- Error Details:", e)
        print("\n----- Common Causes:")
        print("----- Is the MongoDB server running?")
        print("----- Is the IP/hostname in your .env file correct?")
        print("----- Is the port correct?")
        print("----- If using Docker, are the containers on the same network?")
        sys.exit(1)
        
    except ConfigurationError as e:
        print(f"\n----- FAILED: Configuration error in connection string.")
        print("----- Error Details:", e)
        print("----- Check the format of your MONGO_CONNECTION_STRING in the .env file.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n----- FAILED: An unexpected error occurred.")
        print("----- Error Details:", e)
        sys.exit(1)
        
    finally:
        if 'client' in locals() and client:
            client.close()
        
if __name__ == "__main__":
    test_mongo_connection()