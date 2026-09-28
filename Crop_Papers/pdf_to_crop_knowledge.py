# ================================================================
# CROP RESEARCH PAPERS -> MASTER DISEASE KNOWLEDGE JSON
# ================================================================
#
# Folder structure:
#
# F:\Reasearch\Crop_Papers\
#     pdf_to_crop_knowledge.py
#     .env
#     arhar.pdf
#     bajra.pdf
#     cotton.pdf
#     ...
#
# Output:
#
# F:\Reasearch\Crop_Papers\
#     extracted_results\
#         arhar.json
#         bajra.json
#         cotton.json
#         ...
#
#     master_crop_disease_knowledge.json
#
# ================================================================

from pathlib import Path
import os
import json
import time
import random
import re

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ================================================================
# 1. BASIC CONFIGURATION
# ================================================================

# Script jis folder me hai, wahi PDF folder hoga
PDF_FOLDER = Path(__file__).resolve().parent

# Per-PDF JSON backups
OUTPUT_FOLDER = PDF_FOLDER / "extracted_results"

# Final combined JSON
MASTER_JSON = PDF_FOLDER / "master_crop_disease_knowledge.json"

# ------------------------------------------------
# IMPORTANT:
# None = saare PDFs process honge
#
# Example:
# ONLY_PDF = "bajra.pdf"
# ------------------------------------------------
ONLY_PDF = "Masoor.pdf"

# ------------------------------------------------
# Maximum PDFs process karne hain
#
# None = all
# ------------------------------------------------
MAX_PDFS = None

# ------------------------------------------------
# True:
# agar extracted_results/crop.json already hai
# to PDF dobara process nahi hoga
# ------------------------------------------------
SKIP_EXISTING = True


# ================================================================
# 2. LOAD API KEY
# ================================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("\nERROR: GEMINI_API_KEY nahi mili.")
    print("Apni .env file check karo.")
    print("\nExample:")
    print("GEMINI_API_KEY=YOUR_API_KEY")
    raise SystemExit(1)


# ================================================================
# 3. GEMINI CLIENT
# ================================================================

client = genai.Client(api_key=API_KEY)


# ================================================================
# 4. MODEL FALLBACK LIST
# ================================================================
#
# Pehle lightweight model try hoga.
# Agar 503 / overload / temporary failure aaye,
# to next model try hoga.
#
# Ye models tumhari current model list me available the.
# ================================================================

MODEL_CANDIDATES = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]


# ================================================================
# 5. RETRY SETTINGS
# ================================================================

MAX_RETRIES_PER_MODEL = 3

INITIAL_RETRY_DELAY = 15

MAX_RETRY_DELAY = 120


# ================================================================
# 6. HELPER FUNCTIONS
# ================================================================

def clean_model_name(name):
    """
    models/gemini-3.5-flash
    ->
    gemini-3.5-flash
    """
    return name.replace("models/", "").strip()


def normalize_crop_name(filename):
    """
    PDF filename se crop name banata hai.

    Example:
    bajra.pdf -> Bajra
    Wheat.pdf -> Wheat
    moongfali.pdf -> Moongfali
    """

    name = Path(filename).stem

    name = re.sub(r"[_\-]+", " ", name)

    return name.strip().title()


def safe_filename(text):
    """
    JSON filename ke liye safe name.
    """

    text = re.sub(r'[<>:"/\\|?*]', "_", text)

    return text.strip()


def extract_json_from_text(text):
    """
    Agar Gemini JSON ke around ```json ... ``` laga de
    to usko clean karke JSON parse karta hai.
    """

    if not text:
        raise ValueError("Gemini ne empty response diya.")

    text = text.strip()

    # Markdown code fence remove
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    # Direct JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # First { and last }
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError("Gemini response valid JSON nahi tha.")


def save_json(path, data):
    """
    UTF-8 formatted JSON save.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ================================================================
# 7. STRUCTURED JSON SCHEMA
# ================================================================

DISEASE_SCHEMA = {
    "type": "object",

    "properties": {

        "crop": {
            "type": "string"
        },

        "paper_information": {
            "type": "object",
            "properties": {

                "title": {
                    "type": "string"
                },

                "authors": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },

                "year": {
                    "type": "string"
                },

                "journal_or_source": {
                    "type": "string"
                },

                "doi_or_identifier": {
                    "type": "string"
                },

                "abstract_summary": {
                    "type": "string"
                },

                "research_objective": {
                    "type": "string"
                },

                "study_area": {
                    "type": "string"
                },

                "important_findings": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }

            }
        },

        "diseases": {
            "type": "array",

            "items": {
                "type": "object",

                "properties": {

                    "disease_name": {
                        "type": "string"
                    },

                    "alternative_names": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },

                    "disease_type": {
                        "type": "string"
                    },

                    "causal_agent": {
                        "type": "string"
                    },

                    "pathogen_type": {
                        "type": "string"
                    },

                    "affected_plant_parts": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },

                    "symptoms": {
                        "type": "object",
                        "properties": {

                            "general": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            },

                            "early_symptoms": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            },

                            "advanced_symptoms": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            },

                            "visual_field_symptoms": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        }
                    },

                    "identification": {
                        "type": "object",
                        "properties": {

                            "field_identification": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            },

                            "diagnostic_methods": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            },

                            "differential_diagnosis": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        }
                    },

                    "causes_and_risk_factors": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },

                    "favorable_conditions": {
                        "type": "object",
                        "properties": {

                            "temperature": {
                                "type": "string"
                            },

                            "humidity": {
                                "type": "string"
                            },

                            "rainfall": {
                                "type": "string"
                            },

                            "soil_conditions": {
                                "type": "string"
                            },

                            "other_conditions": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        }
                    },

                    "disease_development_and_spread": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },

                    "epidemiology": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },

                    "severity_and_yield_loss": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },

                    "management": {
                        "type": "object",
                        "properties": {

                            "cultural_management": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            },

                            "biological_management": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            },

                            "chemical_management": {
                                "type": "array",

                                "items": {
                                    "type": "object",

                                    "properties": {

                                        "active_ingredient_or_product": {
                                            "type": "string"
                                        },

                                        "dose": {
                                            "type": "string"
                                        },

                                        "concentration": {
                                            "type": "string"
                                        },

                                        "application_method": {
                                            "type": "string"
                                        },

                                        "timing": {
                                            "type": "string"
                                        },

                                        "number_of_applications": {
                                            "type": "string"
                                        },

                                        "result_reported_in_paper": {
                                            "type": "string"
                                        }
                                    }
                                }
                            },

                            "integrated_management": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        }
                    },

                    "resistant_or_tolerant_varieties": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },

                    "susceptible_varieties": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },

                    "prevention": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },

                    "farmer_action_points": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },

                    "warnings_and_limitations": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },

                    "research_findings": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },

                    "source_pages": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                }
            }
        },

        "paper_level_notes": {
            "type": "array",
            "items": {
                "type": "string"
            }
        }
    }
}


# ================================================================
# 8. EXTRACTION PROMPT
# ================================================================

def build_prompt(crop_name):
    return f"""
You are a scientific agricultural research-paper extraction system.

CROP:
{crop_name}

TASK:
Read the ENTIRE attached research paper/PDF carefully.

This is NOT a short summary task.

I am building an agricultural disease knowledge base for farmers.

You must extract ALL useful disease-related information supported by
the paper.

============================================================
READ THE COMPLETE PAPER
============================================================

Read and inspect:

- title
- authors
- abstract
- introduction
- literature review
- materials and methods
- observations
- experiments
- results
- discussion
- conclusion
- tables
- table notes
- figures
- figure captions
- graphs
- images
- supplementary information if present
- references when they contain useful disease information

Do NOT stop after the abstract or conclusion.

============================================================
DISEASE EXTRACTION
============================================================

Extract EVERY disease mentioned in the paper that has meaningful
information.

Do not extract only the main disease.

For each disease identify:

1. Disease name
2. Alternative/local/scientific names
3. Disease type
4. Causal agent
5. Pathogen type
6. Plant parts affected
7. Symptoms
8. Early symptoms
9. Advanced symptoms
10. Visible field symptoms
11. How a farmer can recognize it
12. Laboratory/diagnostic methods
13. Differential diagnosis if available
14. Causes
15. Risk factors
16. Favorable temperature
17. Humidity
18. Rainfall
19. Soil conditions
20. Other environmental conditions
21. Disease development
22. Disease cycle
23. Transmission
24. Spread
25. Epidemiology
26. Severity
27. Yield loss
28. Cultural management
29. Biological management
30. Chemical management
31. Integrated disease management
32. Resistant/tolerant varieties
33. Susceptible varieties
34. Prevention
35. Farmer action points
36. Important research findings
37. Important warnings/limitations

============================================================
CHEMICAL MANAGEMENT
============================================================

Be extremely careful.

Only extract chemical recommendations that are actually supported
by the research paper.

For every chemical/product mentioned, extract if available:

- active ingredient
- product name
- dose
- concentration
- application method
- application timing
- number of applications
- interval
- disease reduction/result reported in paper

DO NOT invent doses.

DO NOT assume a chemical is recommended merely because it is
mentioned in the references.

If the paper does not provide a value, write an empty string.

============================================================
BIOLOGICAL MANAGEMENT
============================================================

Extract:

- organism
- antagonist
- biocontrol agent
- formulation/product
- dose/concentration
- application method
- timing
- results

Only if supported by the paper.

============================================================
FARMER IDENTIFICATION
============================================================

The final knowledge base will later be used like this:

Farmer uploads a plant image
        ↓
AI detects crop/disease
        ↓
Knowledge base searches disease
        ↓
Farmer receives useful information

Therefore, extract practical identification information.

Especially preserve:

- what the farmer sees
- where symptoms appear
- shape/color/pattern
- spots/lesions
- leaf symptoms
- stem symptoms
- root symptoms
- flower symptoms
- fruit/pod/grain symptoms
- progression of symptoms
- signs of pathogen
- differences from similar diseases

============================================================
SCIENTIFIC ACCURACY
============================================================

VERY IMPORTANT:

1. Never invent information.
2. Never hallucinate a disease.
3. Never invent chemical doses.
4. Never invent temperature ranges.
5. Never invent yield-loss values.
6. Never convert speculation into fact.
7. Preserve uncertainty from the paper.
8. If information is not present, use an empty string or empty list.
9. If the paper reports multiple values, preserve them.
10. Do not silently replace paper findings with general agricultural knowledge.

============================================================
PAGE TRACEABILITY
============================================================

Whenever possible, record PDF page numbers in source_pages.

If a disease or recommendation is supported by multiple pages,
include all relevant page numbers.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON according to the supplied schema.

Do not return Markdown.

Do not return explanations outside JSON.

Make the extraction detailed and information-rich.

The goal is to lose as little useful research information as
reasonably possible while remaining faithful to the paper.
"""


# ================================================================
# 9. CHECK AVAILABLE MODELS
# ================================================================

def get_available_models():

    print("\nChecking available Gemini models...\n")

    available = set()

    try:
        for model in client.models.list():

            name = getattr(model, "name", None)

            if name:
                available.add(clean_model_name(name))

    except Exception as e:

        print("WARNING: Model list fetch failed:")
        print(e)

    return available


# ================================================================
# 10. SELECT MODELS THAT ACTUALLY EXIST
# ================================================================

def get_model_order():

    available = get_available_models()

    usable = []

    for model in MODEL_CANDIDATES:

        if model in available:
            usable.append(model)

    print("Models selected for fallback:")

    for model in usable:
        print("  -", model)

    if not usable:

        print("\nERROR: Preferred models available list me nahi mile.")
        print("Run check_models.py again.")
        raise SystemExit(1)

    return usable


# ================================================================
# 11. CALL GEMINI
# ================================================================

def analyze_pdf(pdf_path, crop_name, model_order):

    prompt = build_prompt(crop_name)

    last_error = None

    for model_name in model_order:

        print("\n------------------------------------------------------------")
        print("Trying model:", model_name)
        print("------------------------------------------------------------")

        for attempt in range(1, MAX_RETRIES_PER_MODEL + 1):

            try:

                print(
                    f"Gemini analysis attempt "
                    f"{attempt}/{MAX_RETRIES_PER_MODEL}..."
                )

                # ------------------------------------------------
                # Upload PDF
                # ------------------------------------------------

                print("Uploading PDF...")

                uploaded_file = client.files.upload(
                    file=str(pdf_path)
                )

                print("Upload successful.")

                # ------------------------------------------------
                # Generation config
                # ------------------------------------------------

                config = types.GenerateContentConfig(

                    temperature=0.1,

                    # Detailed output
                    max_output_tokens=65536,

                    response_mime_type="application/json",

                    response_schema=DISEASE_SCHEMA,
                )

                # ------------------------------------------------
                # Generate
                # ------------------------------------------------

                response = client.models.generate_content(

                    model=model_name,

                    contents=[
                        uploaded_file,
                        prompt
                    ],

                    config=config
                )

                # ------------------------------------------------
                # Extract text
                # ------------------------------------------------

                response_text = getattr(
                    response,
                    "text",
                    None
                )

                if not response_text:

                    raise ValueError(
                        "Gemini response me text nahi mila."
                    )

                # ------------------------------------------------
                # Parse JSON
                # ------------------------------------------------

                data = extract_json_from_text(
                    response_text
                )

                # ------------------------------------------------
                # Add processing metadata
                # ------------------------------------------------

                data["_processing_metadata"] = {

                    "source_pdf": pdf_path.name,

                    "crop_detected_from_filename": crop_name,

                    "model_used": model_name,

                    "extracted_at": time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                }

                print("\nSUCCESS!")
                print("Model:", model_name)

                return data

            except Exception as e:

                last_error = e

                error_text = str(e)

                print("\nERROR:")
                print(error_text)

                # ------------------------------------------------
                # Decide whether retry makes sense
                # ------------------------------------------------

                temporary_error = any(
                    code in error_text
                    for code in [
                        "503",
                        "UNAVAILABLE",
                        "500",
                        "INTERNAL",
                        "429",
                        "RESOURCE_EXHAUSTED",
                        "rate_limit",
                        "temporarily"
                    ]
                )

                if temporary_error:

                    if attempt < MAX_RETRIES_PER_MODEL:

                        delay = min(
                            INITIAL_RETRY_DELAY * (2 ** (attempt - 1)),
                            MAX_RETRY_DELAY
                        )

                        # Small random jitter
                        delay += random.randint(0, 5)

                        print(
                            f"\nTemporary server/rate-limit issue."
                        )

                        print(
                            f"Waiting {delay} seconds..."
                        )

                        time.sleep(delay)

                    else:

                        print(
                            "\nModel failed after all retries."
                        )

                else:

                    # Non-temporary error:
                    # next model try karo
                    print(
                        "\nNon-temporary error. "
                        "Trying next model..."
                    )

                    break

        print(
            f"\nSwitching from {model_name} "
            f"to next available model..."
        )

    raise RuntimeError(
        f"All Gemini models failed for {pdf_path.name}.\n"
        f"Last error: {last_error}"
    )


# ================================================================
# 12. FIND PDF FILES
# ================================================================

def find_pdfs():

    pdf_files = sorted(
        PDF_FOLDER.glob("*.pdf"),
        key=lambda p: p.name.lower()
    )

    if ONLY_PDF:

        pdf_files = [
            p for p in pdf_files
            if p.name.lower() == ONLY_PDF.lower()
        ]

    if MAX_PDFS is not None:

        pdf_files = pdf_files[:MAX_PDFS]

    return pdf_files


# ================================================================
# 13. BUILD MASTER JSON
# ================================================================

def build_master_json():

    print("\n")
    print("=" * 80)
    print("BUILDING MASTER JSON")
    print("=" * 80)

    files = sorted(
        OUTPUT_FOLDER.glob("*.json"),
        key=lambda p: p.name.lower()
    )

    crop_records = []

    for json_file in files:

        try:

            data = load_json(json_file)

            crop_records.append(data)

        except Exception as e:

            print(
                f"WARNING: {json_file.name} read nahi hua:"
            )

            print(e)

    master = {

        "knowledge_base": {

            "name": "Crop Disease Research Knowledge Base",

            "description":
                "Research-paper-derived agricultural crop disease "
                "knowledge base.",

            "generated_at":
                time.strftime("%Y-%m-%d %H:%M:%S"),

            "source_folder":
                str(PDF_FOLDER),

            "number_of_processed_papers":
                len(crop_records),

            "crops": crop_records
        }
    }

    save_json(
        MASTER_JSON,
        master
    )

    print("\nMASTER JSON CREATED:")
    print(MASTER_JSON)

    print(
        "\nProcessed paper JSON files:",
        len(crop_records)
    )


# ================================================================
# 14. MAIN
# ================================================================

def main():

    print("\n")
    print("=" * 80)
    print("CROP RESEARCH PAPER -> DISEASE KNOWLEDGE EXTRACTOR")
    print("=" * 80)

    print("\nPDF folder:")
    print(PDF_FOLDER)

    # ------------------------------------------------
    # Find PDFs
    # ------------------------------------------------

    pdf_files = find_pdfs()

    if not pdf_files:

        print("\nERROR: Koi PDF nahi mili.")

        print("\nExpected folder:")
        print(PDF_FOLDER)

        raise SystemExit(1)

    print("\nPDF files found:", len(pdf_files))

    for pdf in pdf_files:
        print(" -", pdf.name)

    # ------------------------------------------------
    # Output folder
    # ------------------------------------------------

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------
    # Get model fallback order
    # ------------------------------------------------

    model_order = get_model_order()

    print("\n")
    print("=" * 80)
    print("PROCESSING PDFs")
    print("=" * 80)

    successful = 0
    skipped = 0
    failed = 0

    failed_files = []

    # ============================================================
    # PROCESS EACH PDF
    # ============================================================

    for index, pdf_path in enumerate(
        pdf_files,
        start=1
    ):

        crop_name = normalize_crop_name(
            pdf_path.name
        )

        output_name = (
            safe_filename(
                pdf_path.stem.lower()
            )
            + ".json"
        )

        output_path = (
            OUTPUT_FOLDER / output_name
        )

        print("\n")
        print("=" * 80)
        print(
            f"[{index}/{len(pdf_files)}] "
            f"{pdf_path.name}"
        )
        print("Crop:", crop_name)
        print("=" * 80)

        # ------------------------------------------------
        # Skip already processed PDF
        # ------------------------------------------------

        if SKIP_EXISTING and output_path.exists():

            try:

                existing = load_json(
                    output_path
                )

                # Check whether it is non-empty
                if existing:

                    print(
                        "\nSKIPPING: "
                        "JSON already exists."
                    )

                    print(
                        "Saved:",
                        output_path
                    )

                    skipped += 1

                    continue

            except Exception:

                print(
                    "\nExisting JSON corrupt/invalid."
                )

                print(
                    "Re-processing PDF..."
                )

        # ------------------------------------------------
        # Analyze
        # ------------------------------------------------

        try:

            data = analyze_pdf(
                pdf_path,
                crop_name,
                model_order
            )

            # ------------------------------------------------
            # Save individual crop JSON
            # ------------------------------------------------

            save_json(
                output_path,
                data
            )

            print("\nSaved:")
            print(output_path)

            successful += 1

            # ------------------------------------------------
            # Delay between PDFs
            # ------------------------------------------------
            #
            # Free-tier rate limits ko respect karne ke liye.
            # ------------------------------------------------

            if index < len(pdf_files):

                delay = 10

                print(
                    f"\nWaiting {delay} seconds "
                    "before next PDF..."
                )

                time.sleep(delay)

        except Exception as e:

            failed += 1

            failed_files.append(
                pdf_path.name
            )

            print("\n")
            print("!" * 80)
            print("FAILED:", pdf_path.name)
            print("!" * 80)
            print(e)

            # Next PDF continue
            continue

    # ============================================================
    # BUILD MASTER JSON
    # ============================================================

    build_master_json()

    # ============================================================
    # FINAL REPORT
    # ============================================================

    print("\n")
    print("=" * 80)
    print("FINAL REPORT")
    print("=" * 80)

    print(
        "Successfully processed:",
        successful
    )

    print(
        "Skipped existing:",
        skipped
    )

    print(
        "Failed:",
        failed
    )

    if failed_files:

        print("\nFailed PDFs:")

        for name in failed_files:

            print(
                " -",
                name
            )

    print("\nOutput folder:")
    print(OUTPUT_FOLDER)

    print("\nMaster JSON:")
    print(MASTER_JSON)

    print("\nDONE.")


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    main()