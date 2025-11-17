import os
import sys
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ConfigurationError

def test_mongo_connection():
    """
    Tests the MongoDB connection using the DB_URI environment variable.
    """
    print("--- MongoDB Connection Test ---")
    
    try:
        # 1. Get the connection string from the environment
        connection_string = os.environ.get('DB_URI')
        
        if not connection_string:
            print("ERROR: DB_URI environment variable is not set.")
            print("Make sure you are running this script inside the Docker container.")
            sys.exit(1)
            
        print(f"Attempting to connect with URI: {connection_string[:20]}...") # Print first 20 chars for privacy

        # 2. Create the client. Set a 5-second timeout.
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        
        # 3. Test the connection by pinging the server
        client.admin.command('ping')
        
        print("\n✅ SUCCESS: Connection to MongoDB is successful!")
        
        # Optional: Print server info
        info = client.server_info()
        print(f"Server Version: {info['version']}")
        
    except ConnectionFailure as e:
        print("\nFAILED: Could not connect to MongoDB.")
        print("Error Details:", e)
        print("\nCommon Causes:")
        print("- Is the MongoDB server running?")
        print("- Is the IP/hostname in your .env file correct?")
        print("- Is the port correct?")
        print("- If using Docker, are the containers on the same network?")
        sys.exit(1)
        
    except ConfigurationError as e:
        print(f"\nFAILED: Configuration error in connection string.")
        print("Error Details:", e)
        print("Check the format of your MONGO_CONNECTION_STRING in the .env file.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nFAILED: An unexpected error occurred.")
        print("Error Details:", e)
        sys.exit(1)
        
    finally:
        # Close the connection if it was opened
        if 'client' in locals() and client:
            client.close()

if __name__ == "__main__":
    test_mongo_connection()