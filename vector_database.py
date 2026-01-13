"""
Vector Database Module - ChromaDB Integration
Stores and retrieves face encodings efficiently using vector similarity search
Replaces the inefficient linear search with semantic vector matching
"""

import chromadb
import numpy as np
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class FaceVectorDatabase:
    """
    Manages face encodings in ChromaDB for efficient similarity search
    Handles face encoding storage, retrieval, and similarity matching
    """
    
    def __init__(self, db_path="./face_vector_db", collection_name="employee_faces"):
        """
        Initialize ChromaDB vector database
        
        Args:
            db_path: Path to store ChromaDB data
            collection_name: Name of the face encoding collection
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(db_path, exist_ok=True)
            
            # Initialize ChromaDB with new API (v0.4.x+)
            try:
                # Try new API first
                self.client = chromadb.PersistentClient(path=db_path)
            except TypeError:
                # Fallback for older versions
                try:
                    from chromadb.config import Settings
                    settings = Settings(
                        persist_directory=db_path,
                        anonymized_telemetry=False
                    )
                    self.client = chromadb.Client(settings)
                except Exception:
                    # Fallback to ephemeral client
                    self.client = chromadb.Client()
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Cosine similarity for face embeddings
            )
            
            self.db_path = db_path
            self.collection_name = collection_name
            logger.info(f"ChromaDB initialized at {db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def add_face_encoding(self, employee_id, name, encoding, metadata=None):
        """
        Add or update face encoding for an employee
        
        Args:
            employee_id: Unique employee identifier
            name: Employee name
            encoding: 128-dim face encoding (numpy array)
            metadata: Additional metadata (department, email, etc.)
        """
        try:
            # Convert numpy array to list for storage
            if isinstance(encoding, np.ndarray):
                encoding_list = encoding.tolist()
            else:
                encoding_list = list(encoding)
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'name': name,
                'added_at': datetime.now().isoformat(),
                'encoding_dim': len(encoding_list)
            })
            
            # Add to collection (ChromaDB handles embeddings)
            self.collection.upsert(
                ids=[employee_id],
                embeddings=[encoding_list],
                metadatas=[metadata],
                documents=[f"{name} - {employee_id}"]  # For search purposes
            )
            
            logger.info(f"Face encoding added for {name} ({employee_id})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding face encoding: {e}")
            return False
    
    def search_similar_faces(self, encoding, top_k=5, threshold=0.6):
        """
        Search for similar face encodings using vector similarity
        
        Args:
            encoding: Query face encoding (numpy array)
            top_k: Number of top matches to return
            threshold: Minimum cosine similarity threshold (0-1)
                      1.0 = identical, 0.0 = completely different
        
        Returns:
            list: [
                {
                    'employee_id': str,
                    'name': str,
                    'similarity': float (0-1),
                    'metadata': dict
                }
            ]
        """
        try:
            if isinstance(encoding, np.ndarray):
                encoding_list = encoding.tolist()
            else:
                encoding_list = list(encoding)
            
            # Query the collection
            results = self.collection.query(
                query_embeddings=[encoding_list],
                n_results=top_k
            )
            
            matches = []
            
            if results and results['ids'] and len(results['ids']) > 0:
                for i, employee_id in enumerate(results['ids'][0]):
                    # ChromaDB returns distances, convert to similarity
                    # For cosine distance: similarity = 1 - distance
                    distance = results['distances'][0][i] if results['distances'] else 1.0
                    similarity = 1 - distance
                    
                    if similarity >= threshold:
                        metadata = results['metadatas'][0][i]
                        matches.append({
                            'employee_id': employee_id,
                            'name': metadata.get('name', 'Unknown'),
                            'similarity': float(similarity),
                            'metadata': metadata
                        })
            
            return sorted(matches, key=lambda x: x['similarity'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error searching similar faces: {e}")
            return []
    
    def get_face_encoding(self, employee_id):
        """
        Retrieve face encoding for a specific employee
        
        Args:
            employee_id: Employee identifier
            
        Returns:
            dict or None: Encoding data or None if not found
        """
        try:
            results = self.collection.get(ids=[employee_id])
            
            if results and results['ids']:
                return {
                    'employee_id': employee_id,
                    'encoding': np.array(results['embeddings'][0]),
                    'metadata': results['metadatas'][0]
                }
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving face encoding: {e}")
            return None
    
    def get_all_encodings(self):
        """
        Retrieve all face encodings from database
        
        Returns:
            list: List of all employee face data
        """
        try:
            results = self.collection.get()
            
            if results and results['ids']:
                all_faces = []
                for i, employee_id in enumerate(results['ids']):
                    all_faces.append({
                        'employee_id': employee_id,
                        'encoding': np.array(results['embeddings'][i]),
                        'metadata': results['metadatas'][i]
                    })
                return all_faces
            return []
            
        except Exception as e:
            logger.error(f"Error retrieving all encodings: {e}")
            return []
    
    def delete_employee(self, employee_id):
        """
        Delete face encoding for an employee
        
        Args:
            employee_id: Employee identifier
        """
        try:
            self.collection.delete(ids=[employee_id])
            logger.info(f"Face encoding deleted for {employee_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting face encoding: {e}")
            return False
    
    def get_collection_stats(self):
        """
        Get statistics about the face encoding collection
        
        Returns:
            dict: Collection statistics
        """
        try:
            count = self.collection.count()
            return {
                'total_employees': count,
                'collection_name': self.collection_name,
                'db_path': self.db_path
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}
    
    def batch_add_encodings(self, encodings_data):
        """
        Add multiple face encodings in batch for better performance
        
        Args:
            encodings_data: List of {
                'employee_id': str,
                'name': str,
                'encoding': array,
                'metadata': dict
            }
        """
        try:
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for data in encodings_data:
                ids.append(data['employee_id'])
                
                if isinstance(data['encoding'], np.ndarray):
                    embeddings.append(data['encoding'].tolist())
                else:
                    embeddings.append(list(data['encoding']))
                
                metadata = data.get('metadata', {})
                metadata.update({
                    'name': data['name'],
                    'added_at': datetime.now().isoformat()
                })
                metadatas.append(metadata)
                documents.append(f"{data['name']} - {data['employee_id']}")
            
            # Batch upsert
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            logger.info(f"Batch added {len(ids)} face encodings")
            return True
            
        except Exception as e:
            logger.error(f"Error in batch add: {e}")
            return False
    
    def persist(self):
        """Persist ChromaDB to disk"""
        try:
            self.client.persist()
            logger.info("ChromaDB persisted to disk")
        except Exception as e:
            logger.error(f"Error persisting ChromaDB: {e}")
    
    def close(self):
        """Close database connection"""
        try:
            if self.client:
                self.client.close()
                logger.info("ChromaDB connection closed")
        except Exception as e:
            logger.error(f"Error closing ChromaDB: {e}")
