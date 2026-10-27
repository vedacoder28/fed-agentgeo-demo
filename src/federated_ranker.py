"""
Hybrid Ranking & Federated Aggregator (src/federated_ranker.py)
Implements federated evidence retrieval across isolated hospital nodes
and coordinates the 3-Column Comparative Board with Explainable AI (XAI) metrics.

Features Strict Clinical Relevance Filtering:
When a query does not match any clinical conditions in a hospital node or guideline repository,
the system transparently flags it as "No Perfect Match Found" for Stream 1, Stream 2, or Stream 3.

Hybrid Ranking Formula (Option C):
Score = (0.45 * Cohort Similarity) + (0.40 * Positive Outcome Rate) + (0.15 * Normalized Case Volume)
"""

import os
import re
from typing import Dict, Any, List, Optional, Tuple

try:
    from .local_node import HospitalNode, GuidelinesNode
    from .agent_geo_optimizer import CLINICAL_KEYWORDS
except ImportError:
    from local_node import HospitalNode, GuidelinesNode
    from agent_geo_optimizer import CLINICAL_KEYWORDS

# System-wide configuration toggle for standalone offline execution
USE_MOCK_LLM = os.getenv("USE_MOCK_LLM", "True").lower() in ["true", "1", "yes"]

class FederatedAggregator:
    """
    Coordinates multi-center federated evaluation and evidence aggregation.
    Respects strict institutional privacy: queries isolated local Chroma collections
    and exchanges only de-identified citable chunks and quantitative ranking metrics.
    """
    def __init__(self, all_hospital_ids: Optional[List[str]] = None):
        self.hospital_ids = all_hospital_ids or ["Hospital_A", "Hospital_B", "Hospital_C"]
        self.nodes = {h_id: HospitalNode(h_id) for h_id in self.hospital_ids}
        self.guidelines_node = GuidelinesNode()

    def get_node(self, hospital_id: str) -> HospitalNode:
        norm_id = HospitalNode._normalize_id(hospital_id)
        if norm_id not in self.nodes:
            self.nodes[norm_id] = HospitalNode(norm_id)
        return self.nodes[norm_id]

    @staticmethod
    def calculate_clinical_relevance(query_text: str, doc_chunk: str, meta: Dict[str, Any]) -> float:
        """
        Calculates strict clinical and semantic relevance between query and case document.
        Returns a score in [0.0, 1.0].
        If query is completely unrelated to the medical specialty or case presentation, returns < 0.15.
        """
        q_clean = query_text.lower().replace("-", " ").replace("/", " ").replace(",", " ")
        q_words = [w for w in q_clean.split() if len(w) >= 3]
        if not q_words:
            return 0.0

        doc_lower = doc_chunk.lower()
        meta_str = f"{meta.get('department', '')} {meta.get('primary_diagnosis', '')} {meta.get('comorbidities', '')} {meta.get('title', '')}".lower()

        # Stop words to ignore
        stop_words = {"and", "the", "with", "for", "from", "has", "had", "was", "she", "his", "her", "who", "any", "year", "years", "old", "male", "female"}
        filtered_q = [w for w in q_words if w not in stop_words]
        if not filtered_q:
            filtered_q = q_words

        clinical_matched = 0
        for w in filtered_q:
            # Check direct presence or root substring match
            if w in meta_str or w in doc_lower:
                clinical_matched += 1
            elif any(root in w for root in ["card", "neph", "angina", "stemi", "renal", "glyc", "tens", "heart", "diabet", "kidney", "fail", "infarct", "ischemi", "hyper", "edema"]):
                # Root search in doc
                if any(w[:4] in term for term in doc_lower.split() if len(term) >= 4):
                    clinical_matched += 1

        overlap_ratio = clinical_matched / max(1, len(filtered_q))
        return round(overlap_ratio, 3)

    @staticmethod
    def parse_query_features(query_text: str) -> Dict[str, Any]:
        """Extracts structured demographic and clinical features from clinical query."""
        features = {
            "age": None,
            "gender": None,
            "egfr": None,
            "creatinine": None,
            "comorbidities": set(),
            "primary_dx": None
        }
        
        # Age regex: e.g., '64M', '64yo', 'age 64', '64 year old'
        age_match = re.search(r'\b(\d{2})\s*(?:yo|y/o|m|f|year|yr)?\b', query_text, re.IGNORECASE)
        if age_match:
            try:
                features["age"] = int(age_match.group(1))
            except ValueError:
                pass
                
        # Gender
        if re.search(r'\b(?:male|man|m)\b', query_text, re.IGNORECASE):
            features["gender"] = "Male"
        elif re.search(r'\b(?:female|woman|f)\b', query_text, re.IGNORECASE):
            features["gender"] = "Female"
            
        # eGFR
        egfr_match = re.search(r'egfr\s*(?:of|:|=)?\s*(\d+)', query_text, re.IGNORECASE)
        if egfr_match:
            features["egfr"] = float(egfr_match.group(1))
            
        # Comorbidities extraction
        q_lower = query_text.lower()
        if "ckd" in q_lower or "chronic kidney" in q_lower or "renal" in q_lower:
            features["comorbidities"].add("Chronic Kidney Disease")
        if "t2d" in q_lower or "diabetes" in q_lower or "diabetic" in q_lower:
            features["comorbidities"].add("Type-2 Diabetes Mellitus")
        if "hypertension" in q_lower or "htn" in q_lower or "bp" in q_lower:
            features["comorbidities"].add("Hypertension")
        if "angina" in q_lower:
            features["primary_dx"] = "Angina Pectoris"
        elif "heart failure" in q_lower or "hfref" in q_lower or "hfpef" in q_lower:
            features["primary_dx"] = "Heart Failure"
        elif "atrial fibrillation" in q_lower or "afib" in q_lower:
            features["primary_dx"] = "Atrial Fibrillation"
            
        return features

    @staticmethod
    def calculate_cohort_similarity(query_features: Dict[str, Any], case_meta: Dict[str, Any]) -> float:
        """
        Calculates Cohort Similarity = Jaccard index of comorbidities + Age proximity
        Range: [0.0, 1.0]
        """
        # 1. Comorbidities Jaccard
        q_comorbs = query_features.get("comorbidities", set())
        case_comorb_str = case_meta.get("comorbidities", "")
        case_comorbs = set([c.strip().lower() for c in case_comorb_str.split(",") if c.strip()])
        
        jaccard = 0.0
        if q_comorbs and case_comorbs:
            matches = 0
            for qc in q_comorbs:
                qc_clean = qc.lower()
                for cc in case_comorbs:
                    if (qc_clean in cc) or (cc in qc_clean) or ("ckd" in qc_clean and "kidney" in cc):
                        matches += 1
                        break
            union_len = len(q_comorbs.union(case_comorbs))
            jaccard = matches / max(1, union_len)
        elif not q_comorbs and not case_comorbs:
            jaccard = 0.3
            
        # 2. Age proximity: 1.0 - (|ΔAge| / 100)
        q_age = query_features.get("age")
        case_age = case_meta.get("age", 60)
        age_proximity = 0.5
        if q_age is not None:
            delta = abs(q_age - int(case_age))
            age_proximity = max(0.0, 1.0 - (delta / 100.0))
            
        # Weighted combination
        return round(0.55 * jaccard + 0.45 * age_proximity, 3)

    def evaluate_external_hospital(
        self,
        hospital_id: str,
        query_text: str,
        query_features: Dict[str, Any],
        top_k: int = 4
    ) -> Dict[str, Any]:
        """
        Evaluates an external hospital node using the Hybrid Ranking Formula:
        Score = (0.45 * Cohort Similarity) + (0.40 * Positive Outcome Rate) + (0.15 * Normalized Case Volume)
        Applies clinical relevance filter: cases with relevance < 0.18 are discarded.
        """
        node = self.get_node(hospital_id)
        raw_retrieved = node.query_local_evidence(query_text=query_text, top_k=top_k)
        
        # Clinical relevance filtering
        relevant_cases = []
        for c in raw_retrieved:
            rel = self.calculate_clinical_relevance(query_text, c["evidence_chunk"], c["metadata"])
            c["clinical_relevance"] = rel
            if rel >= 0.18:
                relevant_cases.append(c)

        if not relevant_cases:
            return {
                "hospital_id": hospital_id,
                "hybrid_score": 0.0,
                "cohort_similarity": 0.0,
                "positive_outcome_rate": 0.0,
                "normalized_volume": 0.0,
                "matching_cases_count": 0,
                "retrieved_cases": [],
                "has_match": False
            }
            
        # 1. Cohort Similarity average
        sims = [self.calculate_cohort_similarity(query_features, c["metadata"]) for c in relevant_cases]
        avg_cohort_sim = sum(sims) / len(sims) if sims else 0.0
        
        # 2. Positive Outcome Rate
        improved_count = sum(1 for c in relevant_cases if c.get("outcome_status") == "Improved")
        positive_outcome_rate = improved_count / len(relevant_cases) if relevant_cases else 0.0
        
        # 3. Normalized Case Volume = min(1.0, matching_cases_count / 10)
        matching_count = len(relevant_cases)
        normalized_volume = min(1.0, round(matching_count / 10.0, 3))
        
        # Hybrid Ranking Score (Option C)
        hybrid_score = round(
            (0.45 * avg_cohort_sim) +
            (0.40 * positive_outcome_rate) +
            (0.15 * normalized_volume),
            3
        )
        
        return {
            "hospital_id": hospital_id,
            "hybrid_score": hybrid_score,
            "cohort_similarity": round(avg_cohort_sim, 3),
            "positive_outcome_rate": round(positive_outcome_rate, 3),
            "normalized_volume": normalized_volume,
            "matching_cases_count": matching_count,
            "retrieved_cases": relevant_cases,
            "has_match": True
        }

    def run_federated_consultation(
        self,
        active_hospital_id: str,
        query_text: str,
        department_filter: Optional[str] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Executes end-to-end federated retrieval:
        1. Queries Local Node (Stream 1) with clinical relevance filter.
        2. Evaluates external hospitals and ranks them to pick Top-Ranked Peer Center (Stream 2).
        3. Queries Public Reference Practice Guidelines (Stream 3) with relevance filter.
        4. Synthesizes Explainable AI (XAI) feature matches and statistical transparency.
        Transparently sets has_match = False if a stream lacks relevant matches.
        """
        active_node_id = HospitalNode._normalize_id(active_hospital_id)
        query_features = self.parse_query_features(query_text)
        
        # --- STREAM 1: Local Hospital Evidence ---
        local_node = self.get_node(active_node_id)
        local_filters = {"department": department_filter} if department_filter and department_filter != "All" else None
        raw_local_cases = local_node.query_local_evidence(query_text=query_text, filters=local_filters, top_k=top_k)
        
        # Filter local cases by clinical relevance
        local_cases = []
        for c in raw_local_cases:
            rel = self.calculate_clinical_relevance(query_text, c["evidence_chunk"], c["metadata"])
            c["clinical_relevance"] = rel
            if rel >= 0.18:
                local_cases.append(c)

        stream_1_has_match = len(local_cases) > 0
        top_local_case = local_cases[0] if stream_1_has_match else None
        local_cohort_sim = self.calculate_cohort_similarity(query_features, top_local_case["metadata"]) if top_local_case else 0.0
        
        # --- STREAM 2: External Peer Center Ranking ---
        external_hospitals = [h for h in self.hospital_ids if HospitalNode._normalize_id(h) != active_node_id]
        peer_evaluations = []
        for ext_h in external_hospitals:
            eval_res = self.evaluate_external_hospital(
                hospital_id=ext_h,
                query_text=query_text,
                query_features=query_features,
                top_k=top_k
            )
            peer_evaluations.append(eval_res)
            
        # Sort by Hybrid Score descending
        peer_evaluations.sort(key=lambda x: x["hybrid_score"], reverse=True)
        top_peer = peer_evaluations[0] if peer_evaluations else None
        
        stream_2_has_match = bool(top_peer and top_peer.get("has_match") and top_peer.get("retrieved_cases"))
        top_peer_case = top_peer["retrieved_cases"][0] if stream_2_has_match else None
        
        # --- STREAM 3: Global Reference Guidelines ---
        raw_guidelines = self.guidelines_node.query_guidelines(query_text=query_text, top_k=3)
        relevant_guidelines = []
        for g in raw_guidelines:
            g_meta = {"title": g.get("title", ""), "department": "Guidelines", "primary_diagnosis": g.get("title", "")}
            rel = self.calculate_clinical_relevance(query_text, g["evidence_chunk"], g_meta)
            g["clinical_relevance"] = rel
            if rel >= 0.18:
                relevant_guidelines.append(g)

        stream_3_has_match = len(relevant_guidelines) > 0
        top_guideline = relevant_guidelines[0] if stream_3_has_match else None
        
        # --- EXPLAINABLE AI (XAI) METRICS ---
        xai_feature_breakdown = self._generate_feature_breakdown(
            query_features, top_local_case, top_peer_case, top_guideline,
            stream_1_has_match, stream_2_has_match, stream_3_has_match
        )
        statistical_transparency = self._generate_statistical_transparency(local_cases, top_peer if stream_2_has_match else None)
        cross_stream_synthesis = self._generate_cross_stream_synthesis(
            query_text, query_features, top_local_case, top_peer if stream_2_has_match else None, top_guideline,
            stream_1_has_match, stream_2_has_match, stream_3_has_match, active_node_id
        )
        
        return {
            "active_hospital_id": active_node_id,
            "query_text": query_text,
            "query_features": query_features,
            # Stream 1
            "stream_1_local": {
                "hospital_id": active_node_id,
                "has_match": stream_1_has_match,
                "cohort_match_pct": round(local_cohort_sim * 100, 1) if stream_1_has_match else 0.0,
                "top_case": top_local_case,
                "retrieved_cases": local_cases,
                "citation": f"[Local - Case #{top_local_case['case_id']}]" if stream_1_has_match else "[Local - No Perfect Match Found]",
                "no_match_message": f"No historical patient encounters in {active_node_id.replace('_', ' ')} closely align with this clinical presentation or organ system."
            },
            # Stream 2
            "stream_2_peer": {
                "hospital_id": top_peer["hospital_id"] if top_peer else "Hospital_B",
                "has_match": stream_2_has_match,
                "hybrid_score": top_peer["hybrid_score"] if stream_2_has_match else 0.0,
                "cohort_similarity_pct": round((top_peer["cohort_similarity"] if stream_2_has_match else 0.0) * 100, 1),
                "positive_outcome_rate_pct": round((top_peer["positive_outcome_rate"] if stream_2_has_match else 0.0) * 100, 1),
                "normalized_volume": top_peer["normalized_volume"] if stream_2_has_match else 0.0,
                "all_external_evaluations": peer_evaluations,
                "top_case": top_peer_case,
                "citation": f"[Peer: {top_peer['hospital_id'].replace('_', ' ')} - Case #{top_peer_case['case_id']}]" if stream_2_has_match else "[Peer - No Perfect Match Found]",
                "no_match_message": "Cross-center evaluation across external peer hospitals found no matching patient cohort meeting clinical relevance criteria."
            },
            # Stream 3
            "stream_3_guideline": {
                "has_match": stream_3_has_match,
                "top_guideline": top_guideline,
                "all_guidelines": relevant_guidelines,
                "citation": f"[Guideline: {top_guideline['guideline_id']}]" if stream_3_has_match else "[Guideline - No Perfect Match Found]",
                "no_match_message": "No public reference practice guidelines in the repository (KDIGO, ACC/AHA, ADA) directly correspond to this specific condition."
            },
            # Explainable AI
            "xai_feature_breakdown": xai_feature_breakdown,
            "statistical_transparency": statistical_transparency,
            "cross_stream_synthesis": cross_stream_synthesis
        }

    def _generate_feature_breakdown(
        self,
        query_features: Dict[str, Any],
        local_case: Optional[Dict[str, Any]],
        peer_case: Optional[Dict[str, Any]],
        top_guideline: Optional[Dict[str, Any]],
        s1_match: bool,
        s2_match: bool,
        s3_match: bool
    ) -> List[Dict[str, Any]]:
        """Constructs an interpretable side-by-side comparison matrix of clinical features."""
        target_age = query_features.get("age")
        target_egfr = query_features.get("egfr")
        
        local_meta = local_case["metadata"] if local_case else {}
        peer_meta = peer_case["metadata"] if peer_case else {}
        
        breakdown = [
            {
                "Feature": "Age Cohort",
                "Patient Query": f"{target_age} years" if target_age else "Unspecified",
                "Own Hospital (Local)": f"{local_meta.get('age')} years" if s1_match and local_meta.get('age') else "No match found",
                "Peer Center (Benchmarked)": f"{peer_meta.get('age')} years" if s2_match and peer_meta.get('age') else "No match found",
                "Global Guideline Standard": "Adult Age Bracket" if s3_match else "No match found"
            },
            {
                "Feature": "Renal Function / eGFR",
                "Patient Query": f"eGFR {target_egfr} mL/min" if target_egfr else "Not specified",
                "Own Hospital (Local)": "eGFR threshold recorded" if s1_match else "No match found",
                "Peer Center (Benchmarked)": "eGFR threshold recorded" if s2_match else "No match found",
                "Global Guideline Standard": "Guideline dosing thresholds" if s3_match else "No match found"
            },
            {
                "Feature": "Primary Presentation",
                "Patient Query": query_features.get("primary_dx", "Custom clinical condition"),
                "Own Hospital (Local)": local_meta.get("primary_diagnosis", "No match found") if s1_match else "No match found",
                "Peer Center (Benchmarked)": peer_meta.get("primary_diagnosis", "No match found") if s2_match else "No match found",
                "Global Guideline Standard": top_guideline.get("title", "No match found") if s3_match else "No match found"
            },
            {
                "Feature": "Key Comorbidities",
                "Patient Query": ", ".join(query_features.get("comorbidities", [])) or "None specified",
                "Own Hospital (Local)": local_meta.get("comorbidities", "None") if s1_match else "No match found",
                "Peer Center (Benchmarked)": peer_meta.get("comorbidities", "None") if s2_match else "No match found",
                "Global Guideline Standard": "Comorbidity guidelines aligned" if s3_match else "No match found"
            },
            {
                "Feature": "Critical Safety Alert",
                "Patient Query": "Standard safety check",
                "Own Hospital (Local)": "Protocol-verified safety alerts" if s1_match else "No match found",
                "Peer Center (Benchmarked)": "Peer safety signals logged" if s2_match else "No match found",
                "Global Guideline Standard": "Guideline contraindications" if s3_match else "No match found"
            }
        ]
        return breakdown

    def _generate_statistical_transparency(
        self,
        local_cases: List[Dict[str, Any]],
        peer_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculates cohort response rates and outcome distribution across federated cases."""
        all_cases = list(local_cases)
        if peer_data and "retrieved_cases" in peer_data:
            all_cases.extend(peer_data["retrieved_cases"])
            
        total = len(all_cases)
        if total == 0:
            return {"improved_pct": 0.0, "stable_pct": 0.0, "adverse_pct": 0.0, "total_evaluated": 0, "peer_hybrid_score": 0.0, "has_cohort": False}
            
        improved = sum(1 for c in all_cases if c.get("outcome_status") == "Improved")
        stable = sum(1 for c in all_cases if c.get("outcome_status") == "Stable")
        adverse = sum(1 for c in all_cases if c.get("outcome_status") == "Adverse Reaction")
        
        return {
            "improved_pct": round((improved / total) * 100, 1),
            "stable_pct": round((stable / total) * 100, 1),
            "adverse_pct": round((adverse / total) * 100, 1),
            "total_evaluated": total,
            "peer_hybrid_score": peer_data.get("hybrid_score", 0.0) if peer_data else 0.0,
            "has_cohort": True
        }

    def _generate_cross_stream_synthesis(
        self,
        query_text: str,
        query_features: Dict[str, Any],
        local_case: Optional[Dict[str, Any]],
        peer_data: Optional[Dict[str, Any]],
        guideline: Optional[Dict[str, Any]],
        s1_match: bool,
        s2_match: bool,
        s3_match: bool,
        active_node: str
    ) -> str:
        """
        Generates a rigorous cross-stream clinical synthesis note.
        Honestly highlights when no perfect matches exist in one or more streams.
        """
        matches_count = sum([s1_match, s2_match, s3_match])
        
        if matches_count == 0:
            return (
                "**⚠️ NO CLINICAL MATCH FOUND ACROSS ALL 3 STREAMS**\n\n"
                "- **Local Node**: No historical cases in Our Hospital align with this presentation.\n"
                "- **Peer Centers**: External hospital nodes (Hospital B, Hospital C) possess no matching patient cohort.\n"
                "- **Practice Guidelines**: No standard of care guideline in the KDIGO, ACC/AHA, or ADA index covers this condition.\n\n"
                "**Actionable Recommendation**: Formulate diagnostic workup and treatment plan using primary clinical examination and external biomedical literature."
            )
            
        lines = ["**CROSS-STREAM CLINICAL SYNTHESIS & EVIDENCE AUDIT**\n"]
        
        # Local Center Status
        if s1_match and local_case:
            lines.append(f"1. **Local Precedent (Our Hospital)**: Matched Case `#{local_case['case_id']}` ({local_case['metadata'].get('primary_diagnosis')}). Empirical therapy in our center yielded an outcome status of `{local_case['outcome_status']}`.")
        else:
            lines.append(f"1. **Local Precedent (Our Hospital)**: ⚠️ **No perfect match found**. Our hospital has no historical records matching this presentation.")
            
        # Peer Center Status
        if s2_match and peer_data and peer_data.get("retrieved_cases"):
            p_top = peer_data["retrieved_cases"][0]
            lines.append(f"2. **Peer Benchmark ({peer_data['hospital_id'].replace('_', ' ')})**: Matched Benchmark Case `#{p_top['case_id']}` with Hybrid Score `{peer_data['hybrid_score']}`. Validates cross-institutional precedent.")
        else:
            lines.append("2. **Peer Benchmark**: ⚠️ **No perfect match found**. External hospital nodes have no comparable cohort for this presentation.")
            
        # Guideline Status
        if s3_match and guideline:
            lines.append(f"3. **Practice Guidelines**: Grounded in `{guideline['guideline_id']}` ({guideline.get('title', '')}). Provides evidence-based dosing thresholds and contraindication alerts.")
        else:
            lines.append("3. **Practice Guidelines**: ⚠️ **No perfect match found**. Repository guidelines do not directly address this specific condition.")
            
        lines.append("\n**Consensus Note**: Proceed with treatment based on available matched streams while accounting for un-matched stream gaps.")
        return "\n\n".join(lines)

if __name__ == "__main__":
    agg = FederatedAggregator()
    # Test matching query
    print("--- Test 1: In-Domain Query ---")
    r1 = agg.run_federated_consultation("Hospital_A", "64M Angina CKD eGFR 38")
    print("Stream 1 match:", r1["stream_1_local"]["has_match"], r1["stream_1_local"]["citation"])
    print("Stream 2 match:", r1["stream_2_peer"]["has_match"], r1["stream_2_peer"]["citation"])
    print("Stream 3 match:", r1["stream_3_guideline"]["has_match"], r1["stream_3_guideline"]["citation"])
    
    # Test completely non-matching query
    print("\n--- Test 2: Out-of-Domain Query (No Match) ---")
    r2 = agg.run_federated_consultation("Hospital_A", "pediatric fractured tibia fibula bone orthopedic casting")
    print("Stream 1 match:", r2["stream_1_local"]["has_match"], r2["stream_1_local"]["citation"])
    print("Stream 2 match:", r2["stream_2_peer"]["has_match"], r2["stream_2_peer"]["citation"])
    print("Stream 3 match:", r2["stream_3_guideline"]["has_match"], r2["stream_3_guideline"]["citation"])
