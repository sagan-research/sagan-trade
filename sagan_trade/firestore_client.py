import os
import pathlib

try:
    import firebase_admin
    from firebase_admin import credentials, firestore

    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False


class SaganFirestore:
    """
    Dynamic fetching and filtering module for the Sagan Firestore database.
    """

    def __init__(self, project_id=None, credential_path=None):
        """
        Initializes the Firestore client.
        If credential_path is not provided, uses standard Google Application Default Credentials.
        """
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT", "sagan-quant")

        try:
            if credential_path and pathlib.Path(credential_path).exists():
                cred = credentials.Certificate(credential_path)
                firebase_admin.initialize_app(cred, {"projectId": self.project_id})
            else:
                # Use Application Default Credentials
                firebase_admin.initialize_app(options={"projectId": self.project_id})
        except ValueError:
            # App already initialized
            pass
        except Exception as e:
            if "DefaultCredentialsError" in str(
                type(e).__name__
            ) or "Could not automatically determine credentials" in str(e):
                print("⚠️ Application Default Credentials not found. Initiating Google Sign-In...")
                self.google_login()
                # Retry initialization
                firebase_admin.initialize_app(options={"projectId": self.project_id})
            else:
                raise e

        self.db = firestore.client()

    @staticmethod
    def google_login():
        """
        Triggers the Google Sign-In flow for Application Default Credentials.
        """
        import subprocess

        print("Launching Google Sign-In in your browser...")
        try:
            subprocess.run(["gcloud", "auth", "application-default", "login"], check=True)
            print("✅ Successfully authenticated with Google!")
        except FileNotFoundError:
            raise RuntimeError(
                "The 'gcloud' CLI is required for Google Sign-In but was not found. Please install the Google Cloud SDK."
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Google Sign-In failed: {e}")

    def get_collection(self, collection_name: str) -> list:
        """
        Retrieves all documents in a specified collection.
        Returns a list of dictionaries including the document id as '_id'.
        """
        docs = self.db.collection(collection_name).stream()
        results = []
        for doc in docs:
            data = doc.to_dict()
            data["_id"] = doc.id
            results.append(data)
        return results

    def query(self, collection_name: str, filters: list) -> list:
        """
        Query a collection with flexible filtering.
        filters: A list of tuples, e.g., [('archetype', '==', 'DARM'), ('scores.aggressive', '>', 50)]
        """
        collection_ref = self.db.collection(collection_name)
        query = collection_ref

        for field, op, value in filters:
            query = query.where(field, op, value)

        docs = query.stream()
        results = []
        for doc in docs:
            data = doc.to_dict()
            data["_id"] = doc.id
            results.append(data)

        return results

    def list_collections(self) -> list:
        """
        Introspect the database and return top-level collection names.
        """
        collections = self.db.collections()
        return [c.id for c in collections]

    def get_document(self, collection_name: str, document_id: str) -> dict:
        """
        Retrieves a single document by ID.
        """
        doc_ref = self.db.collection(collection_name).document(document_id)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            data["_id"] = doc.id
            return data
        return None
