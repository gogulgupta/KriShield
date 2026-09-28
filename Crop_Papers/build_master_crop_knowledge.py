import json
import re
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict

# ============================================================
# POWERFUL MASTER BUILDER
# Merges all extracted crop JSON files WITHOUT using AI.
# It preserves the original extracted information and also
# creates a RAG-friendly disease index for later searching.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "extracted_results"

MASTER_FILE = BASE_DIR / "master_crop_disease_knowledge.json"
RAG_FILE = BASE_DIR / "disease_rag_index.json"

# Files you intentionally do NOT want in the current master.
# "moong" can be added back later simply by removing this name.
EXCLUDED_STEMS = {"moong"}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def clean_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    return str(value)

def normalize_name(value):
    value = clean_text(value).lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()

def recursive_text(value):
    """Convert any JSON structure into searchable text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(recursive_text(x) for x in value)
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            if k.startswith("_"):
                continue
            parts.append(clean_text(k))
            parts.append(recursive_text(v))
        return " ".join(x for x in parts if x)
    return clean_text(value)

def first_nonempty(*values):
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value not in (None, "", [], {}):
            return value
    return ""

def get_disease_list(data):
    """
    Supports the expected schema:
        {"diseases": [...]}
    and also tolerates a few common alternate structures.
    """
    if not isinstance(data, dict):
        return []

    diseases = data.get("diseases")

    if isinstance(diseases, list):
        return diseases

    if isinstance(diseases, dict):
        # If a model ever returned {"Disease A": {...}, ...}
        output = []
        for name, details in diseases.items():
            if isinstance(details, dict):
                item = dict(details)
                item.setdefault("disease_name", name)
                output.append(item)
            else:
                output.append({
                    "disease_name": name,
                    "details": details
                })
        return output

    return []

def count_nonempty(value):
    if value is None:
        return 0
    if isinstance(value, str):
        return 1 if value.strip() else 0
    if isinstance(value, list):
        return sum(count_nonempty(x) for x in value)
    if isinstance(value, dict):
        return sum(count_nonempty(v) for v in value.values())
    return 1

def collect_field_presence(disease):
    """
    Gives a simple completeness map without modifying source content.
    """
    important_fields = [
        "disease_name",
        "alternative_names",
        "disease_type",
        "causal_agent",
        "pathogen_type",
        "affected_plant_parts",
        "symptoms",
        "identification",
        "causes_and_risk_factors",
        "favorable_conditions",
        "disease_development_and_spread",
        "epidemiology",
        "severity_and_yield_loss",
        "management",
        "resistant_or_tolerant_varieties",
        "susceptible_varieties",
        "prevention",
        "farmer_action_points",
        "warnings_and_limitations",
        "research_findings",
        "source_pages",
    ]

    return {
        field: bool(count_nonempty(disease.get(field))) if isinstance(disease, dict) else False
        for field in important_fields
    }

# ------------------------------------------------------------
# Load all JSONs
# ------------------------------------------------------------

if not INPUT_DIR.exists():
    raise SystemExit(f"ERROR: Input folder not found: {INPUT_DIR}")

json_files = sorted(INPUT_DIR.glob("*.json"))

if not json_files:
    raise SystemExit(f"ERROR: No JSON files found in {INPUT_DIR}")

records = []
errors = []
excluded = []

for path in json_files:
    if path.stem.lower() in {x.lower() for x in EXCLUDED_STEMS}:
        excluded.append(path.name)
        continue

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            errors.append({
                "file": path.name,
                "error": "Top-level JSON is not an object"
            })
            continue

        records.append((path, data))

    except Exception as e:
        errors.append({
            "file": path.name,
            "error": str(e)
        })

if not records:
    raise SystemExit("ERROR: No usable JSON files found.")

# ------------------------------------------------------------
# Build rich master
# ------------------------------------------------------------

generated_at = datetime.now(timezone.utc).isoformat()

crops = []
all_diseases = []
disease_lookup = defaultdict(list)

total_diseases = 0
field_counter = Counter()
crop_disease_counts = {}

for path, data in records:
    paper_info = data.get("paper_information", {})
    crop = first_nonempty(
        data.get("crop"),
        data.get("crop_name"),
        path.stem
    )

    diseases = get_disease_list(data)
    disease_count = len(diseases)
    total_diseases += disease_count
    crop_disease_counts[clean_text(crop)] = disease_count

    # Preserve the original JSON exactly inside the master.
    original_data = data

    crop_record = {
        "crop": crop,
        "source_file": path.name,
        "source_pdf": path.stem + ".pdf",
        "paper_information": paper_info,
        "paper_level_notes": data.get("paper_level_notes", ""),
        "disease_count": disease_count,
        "diseases": [],
        "_processing_metadata": data.get("_processing_metadata", {}),
        "_original_json_preserved": original_data,
    }

    for index, disease in enumerate(diseases, start=1):
        if not isinstance(disease, dict):
            disease = {
                "disease_name": clean_text(disease)
            }

        disease_name = first_nonempty(
            disease.get("disease_name"),
            disease.get("name"),
            f"Unnamed disease {index}"
        )

        # Stable, deterministic ID for later RAG/database use.
        base_id = f"{normalize_name(crop)}__{normalize_name(disease_name)}"
        disease_id = base_id

        # If duplicate disease names occur within the same crop,
        # make the ID unique without changing the source content.
        existing_ids = {x["disease_id"] for x in all_diseases}
        if disease_id in existing_ids:
            n = 2
            while f"{base_id}__{n}" in existing_ids:
                n += 1
            disease_id = f"{base_id}__{n}"

        presence = collect_field_presence(disease)

        for field, present in presence.items():
            if present:
                field_counter[field] += 1

        # Preserve the extracted disease object unchanged.
        disease_record = {
            "disease_id": disease_id,
            "crop": crop,
            "source_file": path.name,
            "source_pdf": path.stem + ".pdf",
            "source_paper_title": paper_info.get("title", ""),
            "source_pages": disease.get("source_pages", []),
            "disease_name": disease_name,
            "knowledge": disease,
            "field_presence": presence,
        }

        # Searchable text is derived ONLY from extracted JSON.
        # No new facts are added.
        search_text = recursive_text({
            "crop": crop,
            "paper_information": paper_info,
            "disease": disease
        })

        disease_record["search_text"] = search_text

        crop_record["diseases"].append(disease_record)
        all_diseases.append(disease_record)

        disease_lookup[normalize_name(disease_name)].append(disease_id)

    crops.append(crop_record)

# ------------------------------------------------------------
# Build compact RAG documents
# ------------------------------------------------------------

rag_documents = []

for d in all_diseases:
    k = d["knowledge"]

    rag_documents.append({
        "document_id": d["disease_id"],
        "document_type": "crop_disease_knowledge",
        "crop": d["crop"],
        "disease_name": d["disease_name"],
        "alternative_names": k.get("alternative_names", []),
        "causal_agent": k.get("causal_agent", ""),
        "disease_type": k.get("disease_type", ""),
        "symptoms": k.get("symptoms", {}),
        "identification": k.get("identification", {}),
        "causes_and_risk_factors": k.get("causes_and_risk_factors", {}),
        "favorable_conditions": k.get("favorable_conditions", {}),
        "disease_development_and_spread": k.get("disease_development_and_spread", ""),
        "epidemiology": k.get("epidemiology", ""),
        "severity_and_yield_loss": k.get("severity_and_yield_loss", ""),
        "management": k.get("management", {}),
        "resistant_or_tolerant_varieties": k.get("resistant_or_tolerant_varieties", []),
        "susceptible_varieties": k.get("susceptible_varieties", []),
        "prevention": k.get("prevention", []),
        "farmer_action_points": k.get("farmer_action_points", []),
        "warnings_and_limitations": k.get("warnings_and_limitations", []),
        "research_findings": k.get("research_findings", []),
        "source_pages": k.get("source_pages", []),
        "source_file": d["source_file"],
        "source_pdf": d["source_pdf"],
        "source_paper_title": d["source_paper_title"],
        "search_text": d["search_text"],
    })

# ------------------------------------------------------------
# Duplicate-name report
# ------------------------------------------------------------

duplicate_names = {
    name: ids
    for name, ids in disease_lookup.items()
    if len(ids) > 1
}

# ------------------------------------------------------------
# Final master
# ------------------------------------------------------------

master = {
    "knowledge_base_info": {
        "name": "Crop Disease Research Knowledge Base",
        "version": "2.0",
        "purpose": (
            "Research-grounded crop disease knowledge base for "
            "search, RAG, farmer-assistant development and analysis."
        ),
        "generated_at_utc": generated_at,
        "source_folder": str(INPUT_DIR),
        "ai_used_for_merge": False,
        "important_rule": (
            "This merge step does not invent, summarize, correct, or "
            "replace extracted research content. It preserves the "
            "individual JSON data and adds indexing metadata."
        ),
    },

    "coverage": {
        "papers_loaded": len(records),
        "papers_excluded": excluded,
        "paper_errors": errors,
        "crops": len(crops),
        "total_disease_records": total_diseases,
        "crop_disease_counts": crop_disease_counts,
        "duplicate_disease_name_groups": len(duplicate_names),
    },

    "field_coverage": dict(field_counter),

    # Full structured knowledge, grouped by crop.
    "crops": crops,

    # One record per disease, useful for RAG/vector DB/database ingestion.
    "disease_knowledge": all_diseases,

    # Quick normalized lookup: normalized disease name -> disease IDs.
    "disease_name_index": dict(disease_lookup),

    "quality_control": {
        "duplicate_disease_names": duplicate_names,
        "json_files_with_errors": errors,
        "excluded_files": excluded,
        "note": (
            "Empty fields remain empty when the source extraction did not "
            "provide the information. They are not filled with outside knowledge."
        ),
    },
}

# ------------------------------------------------------------
# Save master
# ------------------------------------------------------------

with MASTER_FILE.open("w", encoding="utf-8") as f:
    json.dump(master, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------
# Save RAG index separately
# ------------------------------------------------------------

rag_index = {
    "knowledge_base": "Crop Disease Research Knowledge Base",
    "version": "2.0",
    "generated_at_utc": generated_at,
    "document_count": len(rag_documents),
    "documents": rag_documents,
}

with RAG_FILE.open("w", encoding="utf-8") as f:
    json.dump(rag_index, f, ensure_ascii=False, indent=2)

# ------------------------------------------------------------
# Console report
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("POWERFUL MASTER JSON BUILDER")
print("=" * 80)

print(f"\nInput folder:")
print(INPUT_DIR)

print(f"\nJSON files detected: {len(json_files)}")
print(f"JSON files included : {len(records)}")
print(f"JSON files excluded : {len(excluded)}")
print(f"JSON files with errors: {len(errors)}")

if excluded:
    print("\nExcluded intentionally:")
    for x in excluded:
        print("  -", x)

print("\nCrops included:")
for crop, count in crop_disease_counts.items():
    print(f"  - {crop}: {count} disease records")

print(f"\nTotal disease records: {total_diseases}")

print("\nMaster created:")
print(MASTER_FILE)

print("\nRAG index created:")
print(RAG_FILE)

if duplicate_names:
    print("\nDuplicate disease-name groups found:")
    for name, ids in duplicate_names.items():
        print(f"  - {name}: {len(ids)} records")

if errors:
    print("\nErrors:")
    for e in errors:
        print(f"  - {e['file']}: {e['error']}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
