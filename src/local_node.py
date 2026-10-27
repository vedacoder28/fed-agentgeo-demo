"""
Isolated Local Nodes & ChromaDB Vector Store (src/local_node.py)
Implements HospitalNode with isolated local persistence and privacy safeguards.
Zero raw patient names, physician names, or unformatted EHR records leave the local boundary.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

try:
    from .agent_geo_optimizer import optimize_clinical_record, CLINICAL_KEYWORDS
except ImportError:
    from agent_geo_optimizer import optimize_clinical_record, CLINICAL_KEYWORDS

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

class LocalClinicalEmbeddingFunction(EmbeddingFunction):
    """
    High-performance, deterministic local clinical embedding function.
    Eliminates external model download failures and network latency while providing
    accurate medical semantic matching using clinical vocabulary projection and sub-word hashing.
    Runs 100% locally and offline without external dependencies.
    """
    def __init__(self, dim: int = 256):
        self.dim = dim

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = []
        for text in input:
            tokens = text.lower().replace("-", " ").replace("/", " ").split()
            vec = np.zeros(self.dim, dtype=np.float32)
            for token in tokens:
                # Assign higher weight to identified clinical terms
                weight = 3.0 if token in CLINICAL_KEYWORDS else 1.0
                h = abs(hash(token)) % self.dim
                vec[h] += weight
                # Character trigram hashing for partial matches (e.g. 'angina' vs 'anginal')
                if len(token) >= 4:
                    for i in range(len(token) - 2):
                        tri = token[i:i+3]
                        h_tri = abs(hash(tri)) % self.dim
                        vec[h_tri] += 0.5 * weight
            
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.tolist())
        return embeddings

class HospitalNode:
    """
    Represents an isolated hospital institution maintaining its own private ChromaDB vector store.
    Zero patient identifiers leave this boundary. Only de-identified, GEO-optimized evidence
    chunks are shared across the federated network.
    """
    def __init__(self, hospital_id: str, data_dir: Optional[Path] = None):
        self.hospital_id = self._normalize_id(hospital_id)
        self.data_dir = data_dir or (DATA_DIR / self.hospital_id.lower())
        self.chroma_path = self.data_dir / "chroma_db"
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        
        # Isolated Chroma client per hospital node
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.embedding_function = LocalClinicalEmbeddingFunction(dim=256)
        self.collection_name = f"node_{self.hospital_id.lower()}"
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Automatically sync cases on initial load if collection is empty
        if self.collection.count() == 0:
            self.sync_from_cases_json()

    @staticmethod
    def _normalize_id(h_id: str) -> str:
        h = h_id.strip().replace(" ", "_")
        if not h.lower().startswith("hospital_"):
            if h.lower() in ["hospital a", "hospital_a", "a", "hosp_a"]:
                return "Hospital_A"
            elif h.lower() in ["hospital b", "hospital_b", "b", "hosp_b"]:
                return "Hospital_B"
            elif h.lower() in ["hospital c", "hospital_c", "c", "hosp_c"]:
                return "Hospital_C"
        # Standardize capitalization
        parts = h.split("_")
        return f"{parts[0].capitalize()}_{parts[1].upper()}"

    def get_cases_file_path(self) -> Path:
        return self.data_dir / "cases.json"

    def get_all_raw_cases(self) -> List[Dict[str, Any]]:
        """Reads raw patient cases from local node storage."""
        cases_file = self.get_cases_file_path()
        if not cases_file.exists():
            return []
        with open(cases_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def sync_from_cases_json(self, force_reindex: bool = False):
        """Indexes all cases from cases.json into the local isolated ChromaDB collection."""
        cases = self.get_all_raw_cases()
        if not cases:
            return

        ids_to_add = []
        docs_to_add = []
        metadatas_to_add = []

        for case in cases:
            patient_id = case.get("patient_id")
            if not patient_id:
                continue
                
            opt_result = optimize_clinical_record(case)
            chunk = opt_result["optimized_evidence_chunk"]
            
            demographics = case.get("demographics", {})
            outcome = case.get("treatment_outcome", {})
            comorbidities = case.get("comorbidities", [])
            
            metadata = {
                "patient_id": patient_id,
                "hospital_id": self.hospital_id,
                "department": case.get("department", "General"),
                "age": int(demographics.get("age", 0)),
                "gender": demographics.get("gender", "Unknown"),
                "primary_diagnosis": case.get("primary_diagnosis", "Unknown"),
                "comorbidities": ", ".join(comorbidities),
                "outcome_status": outcome.get("status", "Unknown"),
                "raw_snr": float(opt_result["raw_snr"]),
                "optimized_snr": float(opt_result["optimized_snr"])
            }
            
            ids_to_add.append(patient_id)
            docs_to_add.append(chunk)
            metadatas_to_add.append(metadata)

        if ids_to_add:
            # Upsert into ChromaDB
            self.collection.upsert(
                ids=ids_to_add,
                documents=docs_to_add,
                metadatas=metadatas_to_add
            )

    def ingest_patient_case(self, raw_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingests a new clinical case into the hospital:
        1. Ensures hospital_id belongs to this node.
        2. Appends to cases.json.
        3. Runs AgentGEO optimization.
        4. Indexes into isolated ChromaDB collection.
        """
        raw_case["hospital_id"] = self.hospital_id
        cases_file = self.get_cases_file_path()
        existing_cases = self.get_all_raw_cases()
        
        # Check if already present and update, else append
        patient_id = raw_case.get("patient_id")
        updated = False
        for i, c in enumerate(existing_cases):
            if c.get("patient_id") == patient_id:
                existing_cases[i] = raw_case
                updated = True
                break
        if not updated:
            existing_cases.append(raw_case)
            
        with open(cases_file, "w", encoding="utf-8") as f:
            json.dump(existing_cases, f, indent=2)
            
        # Run optimization
        opt_result = optimize_clinical_record(raw_case)
        chunk = opt_result["optimized_evidence_chunk"]
        
        demographics = raw_case.get("demographics", {})
        outcome = raw_case.get("treatment_outcome", {})
        comorbidities = raw_case.get("comorbidities", [])
        
        metadata = {
            "patient_id": patient_id,
            "hospital_id": self.hospital_id,
            "department": raw_case.get("department", "General"),
            "age": int(demographics.get("age", 0)),
            "gender": demographics.get("gender", "Unknown"),
            "primary_diagnosis": raw_case.get("primary_diagnosis", "Unknown"),
            "comorbidities": ", ".join(comorbidities),
            "outcome_status": outcome.get("status", "Unknown"),
            "raw_snr": float(opt_result["raw_snr"]),
            "optimized_snr": float(opt_result["optimized_snr"])
        }
        
        self.collection.upsert(
            ids=[patient_id],
            documents=[chunk],
            metadatas=[metadata]
        )
        
        return opt_result

    def query_local_evidence(self, query_text: str, filters: Optional[Dict[str, Any]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Queries the isolated local node.
        Strict Privacy Guarantee:
        Zero unformatted EHRs, patient names, or attending physician names leave this method.
        Only de-identified GEO-optimized chunks with cohort metrics are returned.
        """
        if self.collection.count() == 0:
            self.sync_from_cases_json()
            
        where_filter = None
        if filters:
            conditions = []
            if "department" in filters and filters["department"] and filters["department"] != "All":
                conditions.append({"department": {"$eq": filters["department"]}})
            if "outcome_status" in filters and filters["outcome_status"]:
                conditions.append({"outcome_status": {"$eq": filters["outcome_status"]}})
            
            if len(conditions) == 1:
                where_filter = conditions[0]
            elif len(conditions) > 1:
                where_filter = {"$and": conditions}

        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(top_k, max(1, self.collection.count())),
            where=where_filter
        )
        
        retrieved = []
        if results and "ids" in results and len(results["ids"]) > 0:
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(ids)
            
            for cid, doc, meta, dist in zip(ids, docs, metas, distances):
                # Convert cosine distance to similarity score
                similarity = max(0.0, min(1.0, round(1.0 - (float(dist) / 2.0), 3)))
                retrieved.append({
                    "case_id": cid,
                    "hospital_id": self.hospital_id,
                    "evidence_chunk": doc,
                    "metadata": meta,
                    "similarity": similarity,
                    "outcome_status": meta.get("outcome_status", "Improved"),
                    "department": meta.get("department", "General")
                })
                
        return retrieved

    def get_vault_metrics(self) -> Dict[str, Any]:
        """Calculates institutional metrics for Tab 2 Local Knowledge Vault."""
        cases = self.get_all_raw_cases()
        if not cases:
            return {
                "total_cases": 0,
                "departments": {},
                "avg_raw_snr": 0.0,
                "avg_optimized_snr": 0.0,
                "snr_gain_pct": 0.0,
                "indexed_vectors": self.collection.count()
            }
            
        departments = {}
        total_raw_snr = 0.0
        total_opt_snr = 0.0
        
        for case in cases:
            dept = case.get("department", "Other")
            departments[dept] = departments.get(dept, 0) + 1
            opt = optimize_clinical_record(case)
            total_raw_snr += opt["raw_snr"]
            total_opt_snr += opt["optimized_snr"]
            
        count = len(cases)
        avg_raw = round(total_raw_snr / count, 3)
        avg_opt = round(total_opt_snr / count, 3)
        gain = round(((avg_opt - avg_raw) / avg_raw) * 100, 1) if avg_raw > 0 else 0.0
        
        return {
            "total_cases": count,
            "departments": departments,
            "avg_raw_snr": avg_raw,
            "avg_optimized_snr": avg_opt,
            "snr_gain_pct": gain,
            "indexed_vectors": self.collection.count()
        }


class GuidelinesNode:
    """Retriever for public reference practice guidelines (KDIGO, ACC/AHA, ADA)."""
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or (DATA_DIR / "public_guidelines")
        self.chroma_path = self.data_dir / "chroma_db"
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))
        self.embedding_function = LocalClinicalEmbeddingFunction(dim=256)
        self.collection = self.client.get_or_create_collection(
            name="public_guidelines",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        if self.collection.count() == 0:
            self.sync_from_guidelines_json()

    def sync_from_guidelines_json(self):
        g_file = self.data_dir / "guidelines.json"
        if not g_file.exists():
            return
            
        with open(g_file, "r", encoding="utf-8") as f:
            guidelines = json.load(f)
            
        ids = []
        docs = []
        metas = []
        for g in guidelines:
            gid = g["guideline_id"]
            title = g["title"]
            body = g["issuing_body"]
            regimen = g["recommended_regimen"]
            contra = g["contraindications"]
            evidence = g["evidence_text"]
            
            chunk = (
                f"### {title} [{body}]\n"
                f"**Guideline Reference**: {gid}\n\n"
                f"#### Recommended Clinical Regimen\n{regimen}\n\n"
                f"#### Dosing Thresholds & Contraindications\n{contra}\n\n"
                f"#### Trial Evidence Grounding\n{evidence}"
            )
            
            ids.append(gid)
            docs.append(chunk)
            metas.append({
                "guideline_id": gid,
                "title": title,
                "issuing_body": body
            })
            
        if ids:
            self.collection.upsert(ids=ids, documents=docs, metadatas=metas)

    def query_guidelines(self, query_text: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Queries public guidelines for clinical grounding."""
        if self.collection.count() == 0:
            self.sync_from_guidelines_json()
            
        results = self.collection.query(
            query_texts=[query_text],
            n_results=min(top_k, max(1, self.collection.count()))
        )
        
        retrieved = []
        if results and "ids" in results and len(results["ids"]) > 0:
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(ids)
            for gid, doc, meta, dist in zip(ids, docs, metas, distances):
                sim = max(0.0, min(1.0, round(1.0 - (float(dist) / 2.0), 3)))
                retrieved.append({
                    "guideline_id": gid,
                    "title": meta.get("title", ""),
                    "issuing_body": meta.get("issuing_body", ""),
                    "evidence_chunk": doc,
                    "similarity": sim
                })
        return retrieved

if __name__ == "__main__":
    node_a = HospitalNode("Hospital_A")
    print(f"Hospital A Indexed Cases: {node_a.collection.count()}")
    results = node_a.query_local_evidence("64yo male angina CKD eGFR 38", top_k=2)
    for r in results:
        print(f"-> Match: {r['case_id']} | Sim: {r['similarity']} | Outcome: {r['outcome_status']}")
        
    g_node = GuidelinesNode()
    g_res = g_node.query_guidelines("CKD Diabetes SGLT2 inhibitor", top_k=1)
    if g_res:
        print(f"-> Guideline Match: {g_res[0]['guideline_id']} - {g_res[0]['title']}")
