#!/usr/bin/env python3
"""
Script to add sample classification data for testing analytics
"""

from datetime import datetime, timedelta
import random
from bson.objectid import ObjectId

# Sample medical specialties
SPECIALTIES = [
    ' Gastroenterology',
    ' Cardiovascular / Pulmonary',
    ' Dermatology',
    ' Neurology',
    ' Orthopedic',
    ' Emergency Room Reports',
    ' General Medicine',
    ' Pediatrics - Neonatal',
    ' Radiology',
    ' Surgery'
]

# Sample medical texts
SAMPLE_TEXTS = [
    "Patient presents with severe abdominal pain and nausea for the past 3 days.",
    "Cardiac evaluation shows normal sinus rhythm with no significant abnormalities.",
    "Skin examination reveals multiple erythematous patches on the upper extremities.",
    "Neurological assessment indicates normal motor and sensory function bilaterally.",
    "X-ray shows fracture of the right tibia with displacement requiring surgical intervention.",
    "Emergency room visit for chest pain and shortness of breath.",
    "General physical examination reveals no significant findings.",
    "Pediatric patient with fever and cough for 5 days.",
    "CT scan shows normal brain parenchyma with no mass lesions.",
    "Surgical consultation for appendicitis with acute abdominal pain."
]

def add_sample_data():
    """Add sample classification data to the database"""
    try:
        from app import mongo
        
        print("Adding sample classification data...")
        
        # Get existing users
        users = list(mongo.db.users.find())
        if not users:
            print("No users found. Please create some users first.")
            return
        
        # Generate sample classifications
        sample_classifications = []
        
        for i in range(50):  # Add 50 sample classifications
            # Random user
            user = random.choice(users)
            
            # Random specialty
            specialty = random.choice(SPECIALTIES)
            
            # Random text
            text = random.choice(SAMPLE_TEXTS)
            
            # Random timestamp within last 7 days
            days_ago = random.randint(0, 7)
            hours_ago = random.randint(0, 24)
            timestamp = datetime.utcnow() - timedelta(days=days_ago, hours=hours_ago)
            
            classification = {
                'user_id': user['_id'],
                'input_text': text,
                'predicted_specialty': specialty,
                'timestamp': timestamp
            }
            
            sample_classifications.append(classification)
        
        # Insert the sample data
        if sample_classifications:
            result = mongo.db.classifications.insert_many(sample_classifications)
            print(f"✅ Added {len(result.inserted_ids)} sample classifications")
            
            # Show summary
            total_classifications = mongo.db.classifications.count_documents({})
            total_users = mongo.db.users.count_documents({})
            
            print(f"\nSummary:")
            print(f"- Total classifications: {total_classifications}")
            print(f"- Total users: {total_users}")
            
            # Show specialty distribution
            pipeline = [
                {"$group": {"_id": "$predicted_specialty", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}}
            ]
            specialty_stats = list(mongo.db.classifications.aggregate(pipeline))
            
            print(f"\nSpecialty distribution:")
            for stat in specialty_stats[:5]:
                print(f"- {stat['_id']}: {stat['count']}")
                
        else:
            print("No sample data to add.")
            
    except Exception as e:
        print(f"❌ Error adding sample data: {e}")

if __name__ == "__main__":
    print("=== Sample Data Generator ===\n")
    add_sample_data() 