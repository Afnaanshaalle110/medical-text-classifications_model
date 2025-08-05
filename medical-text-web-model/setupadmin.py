
from pymongo import MongoClient
from werkzeug.security import generate_password_hash

def setup_admin_user():
    client = MongoClient('mongodb://localhost:27017/')  # Replace with your MongoDB URI
    db = client.medical_classifier # Ensure this is the correct database name
    users_collection = db.users

    username = input("Enter admin username: ")
    password = input("Enter admin password: ")
    hashed_password = generate_password_hash(password)

    admin_user = {
        "username": username,
        "password": hashed_password,
        "roles": ["admin"]
    }

    try:
        users_collection.insert_one(admin_user)
        print(f"Admin user '{username}' created successfully.")
    except Exception as e:
        print(f"Error creating admin user: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    setup_admin_user() 