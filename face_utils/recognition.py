# face_utils/recognition.py
import numpy as np

from models import Student, FaceEmbedding, Subject
from sqlalchemy.orm import joinedload
from . import embedder

class FaceMatcher:
    def __init__(self, db, embedder):
        self.db = db
        self.embedder = embedder

    def find_match(self, query_emb, subject_id, threshold):
        """
        Find the best matching student among those in the batch linked to the subject.
        """
        # Get subject to know batch
        subject = Subject.query.get(subject_id)
        if not subject:
            return {'matched': False, 'message': 'Subject not found'}

        # Get all students in the batch
        students = Student.query.filter_by(batch_id=subject.batch_id).all()
        if not students:
            return {'matched': False, 'message': 'No students in this batch'}

        best_sim = -1.0
        best_student = None

        for student in students:
            # Get all embeddings for this student
            embeddings = FaceEmbedding.query.filter_by(student_roll_no=student.roll_no).all()
            if not embeddings:
                continue
            # Compute mean embedding
            embs = [deserialize_embedding(e.embedding_blob) for e in embeddings]
            mean_emb = np.mean(embs, axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 0:
                mean_emb = mean_emb / norm
            # Compute similarity
            sim = float(np.dot(query_emb, mean_emb))
            if sim > best_sim:
                best_sim = sim
                best_student = student

        if best_sim < threshold or best_student is None:
            return {
                'matched': False,
                'roll_no': None,
                'name': None,
                'confidence': best_sim * 100,
                'message': 'No match found'
            }
        else:
            return {
                'matched': True,
                'roll_no': best_student.roll_no,
                'name': best_student.name,
                'confidence': best_sim * 100,
                'message': 'OK'
            }

def deserialize_embedding(data):
    return np.frombuffer(data, dtype=np.float32)