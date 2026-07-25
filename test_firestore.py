import sagan_trade

def test_firestore():
    print("Testing Firestore Integration...")
    # Initialize using Application Default Credentials
    db = sagan_trade.SaganFirestore()
    
    # Introspect schema
    collections = db.list_collections()
    print("Available Collections:", collections)
    
    # Query 'results' collection (using data we know exists in sagan-quant)
    docs = db.query('results', filters=[('archetype', '==', 'DARM')])
    print(f"Found {len(docs)} documents with archetype == 'DARM'")
    if docs:
        print("Sample Doc:")
        print(docs[0])

if __name__ == "__main__":
    test_firestore()
