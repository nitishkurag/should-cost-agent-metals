from groq import Groq
import os
import re
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import requests

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
KB_PATH = os.getenv("KB_PATH", "kb.xlsx")

st.set_page_config(page_title="Should-Cost Agent — Fabricated Metal Parts", layout="wide")

# ------------------------------------------------------------------
# LOAD KB (internal only — never shown to user)
# ------------------------------------------------------------------
@st.cache_data
def load_kb(path: str):
    return {
        "rm": pd.read_excel(path, sheet_name="RM_Rates"),
        "labor": pd.read_excel(path, sheet_name="Labor_Rates"),
        "machine": pd.read_excel(path, sheet_name="Machine_Rates"),
        "logistics": pd.read_excel(path, sheet_name="Logistics_Rates"),
        "overhead": pd.read_excel(path, sheet_name="Overhead_Margin"),
        "fx": pd.read_excel(path, sheet_name="Currency_Rates"),
        "country_ccy": pd.read_excel(path, sheet_name="Country_Currency"),
    }

try:
    KB = load_kb(KB_PATH)
except Exception as e:
    st.error(f"Knowledge base not found: {e}")
    st.stop()

REGIONS = sorted(KB["rm"]["Region"].dropna().unique().tolist())
GRADES = sorted(KB["rm"]["Grade"].dropna().unique().tolist())

# Human-friendly material family for display (Sources pane)
def material_family(grade: str) -> str:
    g = grade.lower()
    if "stainless" in g:
        return "Stainless Steel"
    if "alumin" in g:
        return "Aluminum"
    if "steel" in g:
        return "Steel"
    return grade

# ------------------------------------------------------------------
# CURRENCY
# ------------------------------------------------------------------
def fx_rate(from_ccy: str, to_ccy: str = "USD") -> float:
    if from_ccy == to_ccy:
        return 1.0
    df = KB["fx"]
    fwd = df[(df["From_Currency"] == from_ccy) & (df["To_Currency"] == to_ccy)]
    if not fwd.empty:
        return float(fwd["Rate"].iloc[0])
    rev = df[(df["From_Currency"] == to_ccy) & (df["To_Currency"] == from_ccy)]
    if not rev.empty:
        return 1.0 / float(rev["Rate"].iloc[0])
    if to_ccy == "USD":
        return 1.0 / fx_rate("USD", from_ccy)
    if from_ccy == "USD":
        return fx_rate("USD", to_ccy)
    return fx_rate("USD", to_ccy) / fx_rate("USD", from_ccy)

def local_to_usd_per_kg(price_local: float, currency: str, unit_basis: str) -> float:
    usd = price_local * fx_rate(currency, "USD")
    ub = str(unit_basis).lower()
    if "tonne" in ub or "/t" in ub or "ton" in ub:
        return usd / 1000.0
    if "lb" in ub:
        return usd / 0.453592
    return usd

def default_currency(country: str) -> str:
    df = KB["country_ccy"]
    row = df[df["Country"] == country]
    return str(row["Default_Currency"].iloc[0]) if not row.empty else "USD"

# ------------------------------------------------------------------
# KB LOOKUPS (internal)
# ------------------------------------------------------------------
def get_rm_row(grade: str, region: str) -> dict:
    df = KB["rm"]
    row = df[(df["Grade"] == grade) & (df["Region"] == region)]
    if row.empty:
        row = df[df["Grade"] == grade].head(1)
    r = row.iloc[0]
    return {
        "price_local": float(r["Price_Local"]),
        "currency": str(r["Currency"]),
        "unit_basis": str(r["Unit_Basis"]),
        "price_usd_per_kg": local_to_usd_per_kg(float(r["Price_Local"]), str(r["Currency"]), str(r["Unit_Basis"])),
        "source": str(r["Source"]),
        "effective_date": str(r["Effective_Date"]),
    }

def get_labor_rate_usd_hr(region: str) -> dict:
    df = KB["labor"]
    row = df[df["Region"] == region]
    if row.empty:
        row = df.head(1)
    r = row.iloc[0]
    return {
        "usd_per_hr": float(r["Rate_Local"]) * fx_rate(str(r["Currency"]), "USD"),
        "local": float(r["Rate_Local"]),
        "currency": str(r["Currency"]),
        "source": str(r["Source"]),
    }

def get_machine_rate_usd_hr(region: str, process: str) -> dict:
    df = KB["machine"]
    row = df[(df["Region"] == region) & (df["Process"] == process)]
    if row.empty:
        row = df[(df["Process"] == process)]
    if row.empty:
        return {"usd_per_hr": 0.0, "local": 0.0, "currency": "USD", "source": "n/a"}
    r = row.iloc[0]
    return {
        "usd_per_hr": float(r["Rate_Local"]) * fx_rate(str(r["Currency"]), "USD"),
        "local": float(r["Rate_Local"]),
        "currency": str(r["Currency"]),
        "source": str(r["Source"]),
    }

def get_logistics_rate_usd(region: str) -> dict:
    df = KB["logistics"]
    row = df[df["Region"] == region]
    if row.empty:
        row = df.head(1)
    r = row.iloc[0]
    return {
        "freight_usd_per_kg": float(r["Freight_per_kg_Local"]) * fx_rate(str(r["Currency"]), "USD"),
        "packaging_usd": float(r["Packaging_Local"]) * fx_rate(str(r["Currency"]), "USD"),
        "currency": str(r["Currency"]),
        "source": str(r["Source"]),
    }

def get_overhead_margin(region: str, industry: str) -> dict:
    df = KB["overhead"]
    row = df[(df["Region"] == region) & (df["Industry"] == industry)]
    if row.empty:
        row = df[df["Region"] == region]
    if row.empty:
        row = df.head(1)
    r = row.iloc[0]
    return {
        "overhead_pct": float(r["Overhead_Pct"]) / 100.0,
        "margin_pct": float(r["Margin_Pct"]) / 100.0,
        "source": str(r["Source"]),
    }

# ------------------------------------------------------------------
# MARKET INDEX REFERENCES (for forecast, cited in UI)
# ------------------------------------------------------------------
FORECAST_DRIVERS = {
    "Raw Material": {
        "index": "PPI: Steel mill products (FRED WPU1017) / LME Aluminium cash",
        "annual_growth": 0.06,
        "note": "Steel PPI +6% YoY baseline; aluminium tracks LME forward curve",
    },
    "Labor": {
        "index": "BLS Employment Cost Index: manufacturing wages & salaries",
        "annual_growth": 0.035,
        "note": "ECI manufacturing +3.2% 12-month (Jun-2026); 3.5% assumed",
    },
    "Electricity": {
        "index": "EIA Industrial electricity price",
        "annual_growth": 0.06,
        "note": "EIA industrial average 8.51 ¢/kWh (2025), +4.7% YoY; 6% assumed",
    },
    "Logistics": {
        "index": "BLS PPI: General freight trucking, long-distance TL",
        "annual_growth": 0.10,
        "note": "PPI freight +24% YoY (Aug-2026); 10% assumed for normalization",
    },
    "Overhead & Margin": {
        "index": "Supplier overhead & margin — negotiated",
        "annual_growth": 0.02,
        "note": "No public index; 2% assumed as cost inflation pass-through",
    },
}

# ------------------------------------------------------------------
# NARRATIVE SUMMARY (deterministic — no external model)
# ------------------------------------------------------------------
def build_narrative(model: dict) -> str:
    total = model["total"]
    levers = model["levers"]
    rm = levers["Raw material"]
    conv = levers["Conversion (labor + machine)"]
    oh = levers["Overhead"]
    mg = levers["Margin"]

    largest_name = max(levers, key=lambda k: levers[k]["value"])
    largest_val = levers[largest_name]["value"]
    largest_pct = largest_val / total if total else 0.0

    rm_pct = rm["value"] / total if total else 0.0
    oh_mg_pct = (oh["value"] + mg["value"]) / total if total else 0.0

    return (
        f"**Should-cost: ${total:.2f} per unit** ({model['region']}, {model['material']}).\n\n"
        f"- The largest cost driver is **{largest_name}** at ${largest_val:.2f} "
        f"({largest_pct:.0%} of the total).\n"
        f"- **Raw material** contributes ${rm['value']:.2f} ({rm_pct:.0%}), based on "
        f"${rm['detail']['price_usd_per_kg']:.2f}/kg for {model['material']} in {model['region']}, "
        f"a gross weight of {rm['detail']['gross_kg']:.2f} kg "
        f"(incl. {rm['detail']['scrap_pct']:.0%} scrap).\n"
        f"- **Conversion** (labor + machine) contributes ${conv['value']:.2f}, reflecting the selected "
        f"operations at regional machine and labour rates.\n"
        f"- **Overhead + margin** together account for {oh_mg_pct:.0%} of the total, "
        f"in line with regional/industry benchmarks for {model['industry']}.\n"
        f"- The 12-month forecast applies published index growth rates (steel/aluminium PPI, ECI wages, "
        f"EIA electricity, freight PPI) to each driver rather than a flat escalation.\n\n"
        f"**What to validate with the supplier:** the actual scrap rate, cycle times for the selected "
        f"operations, and the freight lane / packaging spec — these are the assumptions most likely to "
        f"shift the should-cost."
    )

# ------------------------------------------------------------------
# COST ENGINE
# ------------------------------------------------------------------
def compute_should_cost_from_kb(g: dict) -> dict:
    region = g["region"]
    industry = g.get("industry", "General fabrication")
    if industry not in KB["overhead"]["Industry"].unique():
        industry = KB["overhead"]["Industry"].iloc[0]

    rm = get_rm_row(g["material"], region)
    weight = float(g.get("weight_kg", 1.20))
    scrap = float(g.get("scrap_pct", 0.12))
    gross = weight * (1 + scrap)
    rm_cost = gross * rm["price_usd_per_kg"]

    labor = get_labor_rate_usd_hr(region)
    conv_lines = []
    procs = g.get("processes_applicable", ["cutting", "bending"])

    if "cutting" in procs:
        m = get_machine_rate_usd_hr(region, "Laser cutting")
        hrs = float(g.get("cut_perimeter_mm", 1800)) / 1000.0 / 6.0
        conv_lines.append({"process": "Laser cutting", "cost": hrs * m["usd_per_hr"], "hours": hrs, "machine_usd_hr": m["usd_per_hr"], "source": m["source"]})
    if "bending" in procs:
        m = get_machine_rate_usd_hr(region, "Hydraulic Press")
        hrs = float(g.get("n_bends", 6)) * 0.05
        conv_lines.append({"process": "Bending (hydraulic press)", "cost": hrs * m["usd_per_hr"],
                           "hours": hrs, "machine_usd_hr": m["usd_per_hr"], "source": m["source"]})    
    if "welding" in procs:
        m = get_machine_rate_usd_hr(region, "Welding")
        hrs = float(g.get("weld_length_mm", 420)) / 1000.0 / 0.3
        conv_lines.append({"process": "Welding", "cost": hrs * m["usd_per_hr"], "hours": hrs, "machine_usd_hr": m["usd_per_hr"], "source": m["source"]})
    if "machining" in procs:
        m = get_machine_rate_usd_hr(region, "CNC Machining")
        hrs = 0.25
        conv_lines.append({"process": "CNC Machining", "cost": hrs * m["usd_per_hr"], "hours": hrs, "machine_usd_hr": m["usd_per_hr"], "source": m["source"]})
    if "surface_treatment" in procs:
        m = get_machine_rate_usd_hr(region, "Hydraulic Press")
        hrs = float(g.get("surface_area_m2", 0.18)) * 0.2
        conv_lines.append({"process": "Surface treatment", "cost": hrs * m["usd_per_hr"], "hours": hrs, "machine_usd_hr": m["usd_per_hr"], "source": m["source"]})

    conv_total = sum(x["cost"] for x in conv_lines)

    energy_rate_kwh = 0.0851
    energy_kwh = conv_total / max(labor["usd_per_hr"], 1) * 2.5
    energy_cost = energy_kwh * energy_rate_kwh

    om = get_overhead_margin(region, industry)
    oh_total = (rm_cost + conv_total + energy_cost) * om["overhead_pct"]
    subtotal = rm_cost + conv_total + energy_cost + oh_total

    logi = get_logistics_rate_usd(region)
    logistics = weight * logi["freight_usd_per_kg"] + logi["packaging_usd"]

    pre_margin = subtotal + logistics
    margin = pre_margin * om["margin_pct"]
    total = pre_margin + margin

    levers = {
        "Raw material": {"value": rm_cost, "detail": {
            "grade": g["material"], "region": region, "price_usd_per_kg": rm["price_usd_per_kg"],
            "price_local": rm["price_local"], "currency": rm["currency"], "unit_basis": rm["unit_basis"],
            "source": rm["source"], "effective_date": rm["effective_date"],
            "weight_kg": weight, "scrap_pct": scrap, "gross_kg": gross}},
        "Conversion (labor + machine)": {"value": conv_total, "detail": {
            "lines": conv_lines, "labor_usd_per_hr": labor["usd_per_hr"], "labor_source": labor["source"]}},
        "Electricity": {"value": energy_cost, "detail": {
            "rate_usd_per_kwh": energy_rate_kwh, "kwh_estimated": round(energy_kwh, 3),
            "source": "EIA Electric Power Monthly: industrial average 8.51 ¢/kWh (2025)"}},
        "Overhead": {"value": oh_total, "detail": {
            "pct": om["overhead_pct"], "industry": industry, "source": om["source"]}},
        "Logistics": {"value": logistics, "detail": {
            "freight_usd_per_kg": logi["freight_usd_per_kg"], "packaging_usd": logi["packaging_usd"], "source": logi["source"]}},
        "Margin": {"value": margin, "detail": {
            "pct": om["margin_pct"], "source": om["source"]}},
    }
    return {
        "total": total, "levers": levers, "region": region, "industry": industry,
        "material": g["material"], "weight_kg": weight, "volume": int(g.get("volume", 5000)),
        "display_ccy": g.get("display_ccy", default_currency(region)),
        "processes_applicable": procs,
    }

# ------------------------------------------------------------------
# LLM-FIRST SHOULD-COST BUILDER (Groq structured outputs)
# ------------------------------------------------------------------
SHOULD_COST_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "region": {"type": "string"},
        "material_grade": {"type": "string"},
        "rm_price_usd_per_kg": {"type": "number"},
        "rm_currency": {"type": "string"},
        "rm_unit_basis": {"type": "string"},
        "rm_source": {"type": "string"},
        "rm_price_source_date": {
            "type": "string",
            "description": "Date the raw material price is as of, in YYYY-MM-DD or YYYY-MM format. Use your best knowledge; if uncertain, use '2024-06'."
        },
        "processes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "machine_rate_usd_hr": {"type": "number"},
                    "hours": {"type": "number"},
                    "source": {"type": "string"},
                },
                "required": ["name", "machine_rate_usd_hr", "hours", "source"],
            },
        },
        "labor_rate_usd_hr": {"type": "number"},
        "labor_source": {"type": "string"},
        "electricity_rate_usd_per_kwh": {"type": "number"},
        "electricity_source": {"type": "string"},
        "overhead_pct": {"type": "number", "description": "Overhead as a decimal fraction, e.g. 0.15 for 15%"},
        "overhead_source": {"type": "string"},
        "logistics_freight_usd_per_kg": {"type": "number"},
        "logistics_packaging_usd": {"type": "number"},
        "logistics_source": {"type": "string"},
        "margin_pct":   {"type": "number", "description": "Margin as a decimal fraction, e.g. 0.10 for 10%"},
        "margin_source": {"type": "string"},
        "forecast_drivers": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "driver": {"type": "string"},
                    "annual_growth": {"type": "number"},
                    "index_reference": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["driver", "annual_growth", "index_reference", "note"],
            },
        },
    },
    "required": [
        "region", "material_grade", "rm_price_usd_per_kg", "rm_currency",
        "rm_unit_basis", "rm_source", "processes", "labor_rate_usd_hr",
        "labor_source", "electricity_rate_usd_per_kwh", "electricity_source",
        "overhead_pct", "overhead_source", "logistics_freight_usd_per_kg",
        "logistics_packaging_usd", "logistics_source", "margin_pct",
        "margin_source", "forecast_drivers", "rm_price_source_date",
    ],
}

def call_groq_price_search(grade: str, region: str) -> str:
    """Step 1: Use Groq browser search to fetch the latest market price for the material."""
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing")

    client = Groq(api_key=api_key, timeout=90.0, max_retries=1)
    query = (
        f"Use web search to find the current market price for {grade} in {region}. "
        f"Search sources: MEPS, Platts, Fastmarkets, USGS, LME. "
        f"Return ONE short paragraph containing: (1) price value, (2) unit (USD/kg, USD/lb, USD/tonne), "
        f"(3) currency, (4) source name, (5) date. "
        f"Do not call further tools; do not open additional pages; return the summary now."
    )

    try:
        r = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": query}],
            temperature=0.0,
            max_completion_tokens=2048,
            tools=[{"type": "browser_search"}],
        )
        content = (r.choices[0].message.content or "").strip()
        if not content or len(content) < 40:
            raise ValueError("Empty browser-search summary")
        return content
    except Exception as e:
        raise RuntimeError(str(e).split("\n")[0])

def call_groq_should_cost(part_description: str, part_name: str, region: str, grade: str, price_reference: str = "") -> dict:
    """Ask Groq for a full should-cost rate sheet as strict JSON."""
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY missing")

    system_prompt = (
        "You are a senior procurement cost analyst for fabricated metal parts. "
        "Given a part description, return realistic industry-standard cost rates "
        "for the specified region and material. Use the latest published indices "
        "you know (e.g., MEPS steel, LME aluminium, BLS ECI, EIA industrial "
        "electricity, freight PPI). Each source field must name a specific public "
        "index or benchmark—never an internal file. Return only the JSON object "
        "matching the provided schema."
        "If a LIVE PRICE REFERENCE is provided, it overrides your training-data "
        "estimate for raw material price. Use it directly and cite its source.\n"
    )
    user_prompt = (
        f"Part name: {part_name}\n"
        f"Part description: {part_description}\n"
        f"Region: {region}\n"
        f"Material grade: {grade}\n\n"
    )
    if price_reference:
        user_prompt += (
            f"LIVE PRICE REFERENCE (from web search, use this as the primary "
            f"basis for rm_price_usd_per_kg — do not use your training-data price):\n"
            f"{price_reference}\n\n"
            f"If the reference price is in a different unit (lb, tonne), convert it "
            f"to USD/kg before filling rm_price_usd_per_kg. "
            f"Set rm_source to the source name and date from the reference.\n\n"
        )
    user_prompt += (
        "Determine which manufacturing processes apply from the description "
        "(cutting, bending, welding, machining, surface treatment). "
        "Return realistic machine rates, process hours, labor rate, electricity "
        "rate, overhead %, logistics costs, margin %, and 12-month forecast "
        "drivers with annual growth rates and index references."
    )

    client = Groq(api_key=api_key, timeout=45.0, max_retries=0)
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ShouldCostModel",
                "strict": True,
                "schema": SHOULD_COST_SCHEMA,
            },
        },
        temperature=0.0,
        max_completion_tokens=4096,
    )
    return json.loads(response.choices[0].message.content)


def validate_llm_rates(data: dict) -> None:
    """Reject implausible rates and normalize percent fields to decimal."""
    rm = data["rm_price_usd_per_kg"]
    if not (0.5 <= rm <= 50.0):
        raise ValueError(f"Raw material rate out of band: {rm}")

    # Normalize percentages: accept 15 or 0.15 and convert to decimal
    for key in ("overhead_pct", "margin_pct"):
        v = data[key]
        if v > 1.0:            # LLM sent "15" meaning 15%
            v = v / 100.0
        if not (0.0 <= v <= 0.6):
            raise ValueError(f"{key} out of band: {data[key]}")
        data[key] = v          # write back normalized value

    if not (0.0 <= data["labor_rate_usd_hr"] <= 200.0):
        raise ValueError(f"Labor rate out of band: {data['labor_rate_usd_hr']}")


from datetime import datetime, timedelta

def parse_price_date(date_str: str) -> datetime | None:
    """Parse LLM-returned date string. Returns None if unparseable."""
    if not date_str:
        return None
    s = date_str.strip().replace("/", "-")
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    m = re.search(r"(20\d{2})", s)
    if m:
        return datetime(int(m.group(1)), 6, 1)
    return None


def is_stale(price_date: datetime | None, max_age_days: int = 90) -> bool:
    """True if price_date is None or older than max_age_days."""
    if price_date is None:
        return True
    return (datetime.now() - price_date).days > max_age_days

def assemble_model_from_llm(data: dict, g: dict) -> dict:
    """Reuse your existing lever structure so downstream panes stay unchanged."""
    weight = float(g.get("weight_kg", 1.20))
    scrap = float(g.get("scrap_pct", 0.12))
    gross = weight * (1 + scrap)
    rm_cost = gross * data["rm_price_usd_per_kg"]

    conv_lines = []
    for p in data["processes"]:
        cost = p["machine_rate_usd_hr"] * p["hours"]
        conv_lines.append({
            "process": p["name"],
            "cost": cost,
            "hours": p["hours"],
            "machine_usd_hr": p["machine_rate_usd_hr"],
            "source": p["source"],
        })
    conv_total = sum(x["cost"] for x in conv_lines)

    energy_kwh = conv_total / max(data["labor_rate_usd_hr"], 1) * 2.5
    energy_cost = energy_kwh * data["electricity_rate_usd_per_kwh"]

    oh_total = (rm_cost + conv_total + energy_cost) * data["overhead_pct"]
    subtotal = rm_cost + conv_total + energy_cost + oh_total

    logistics = (
        weight * data["logistics_freight_usd_per_kg"]
        + data["logistics_packaging_usd"]
    )
    pre_margin = subtotal + logistics
    margin = pre_margin * data["margin_pct"]
    total = pre_margin + margin

    levers = {
        "Raw material": {"value": rm_cost, "detail": {
            "grade": data["material_grade"],
            "region": data["region"],
            "price_usd_per_kg": data["rm_price_usd_per_kg"],
            "price_local": data["rm_price_usd_per_kg"],
            "currency": data["rm_currency"],
            "unit_basis": data["rm_unit_basis"],
            "source": data["rm_source"],
            "effective_date": "LLM-sourced",
            "weight_kg": weight, "scrap_pct": scrap, "gross_kg": gross,
        }},
        "Conversion (labor + machine)": {"value": conv_total, "detail": {
            "lines": conv_lines,
            "labor_usd_per_hr": data["labor_rate_usd_hr"],
            "labor_source": data["labor_source"],
        }},
        "Electricity": {"value": energy_cost, "detail": {
            "rate_usd_per_kwh": data["electricity_rate_usd_per_kwh"],
            "kwh_estimated": round(energy_kwh, 3),
            "source": data["electricity_source"],
        }},
        "Overhead": {"value": oh_total, "detail": {
            "pct": data["overhead_pct"],
            "industry": g.get("industry", "General fabrication"),
            "source": data["overhead_source"],
        }},
        "Logistics": {"value": logistics, "detail": {
            "freight_usd_per_kg": data["logistics_freight_usd_per_kg"],
            "packaging_usd": data["logistics_packaging_usd"],
            "source": data["logistics_source"],
        }},
        "Margin": {"value": margin, "detail": {
            "pct": data["margin_pct"],
            "source": data["margin_source"],
        }},
    }
    return {
        "total": total,
        "levers": levers,
        "region": data["region"],
        "industry": g.get("industry", "General fabrication"),
        "material": data["material_grade"],
        "weight_kg": weight,
        "volume": int(g.get("volume", 5000)),
        "display_ccy": g.get("display_ccy", "USD"),
        "processes_applicable": [p["name"] for p in data["processes"]],
        "llm_forecast_drivers": data["forecast_drivers"],
        "source_mode": "LLM (Groq gpt-oss-120b)",
    }

def search_price_change_pct(material_grade: str, source_date: datetime) -> tuple[float | None, str]:
    """
    Search for the % change in material price from source_date to now.
    Returns (pct_as_decimal, raw_snippet) or (None, error_message).
    """
    g = material_grade.lower()
    if "stainless" in g:
        label = "stainless steel"
    elif "alumin" in g:
        label = "aluminum"
    else:
        label = "steel"

    date_str = source_date.strftime("%B %Y")
    # More targeted query: ask specifically for the % change between two periods
    query = (
        f"\"{label}\" price \"percent change\" OR \"% change\" "
        f"from {date_str} to 2026 site:worldbank.org OR site:steelmarketupdate.com"
    )

    api_key = st.secrets.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return None, "TAVILY_API_KEY not configured in secrets"

    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={
                "query": query,
                "search_depth": "advanced",   # advanced gives better snippets
                "max_results": 5,
                "include_answer": True,
            },
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()

        answer = (data.get("answer") or "").strip()
        snippets = " ".join(
            (item.get("content") or "") for item in (data.get("results") or [])
        )
        combined = (answer + " " + snippets).lower()

        # Require BOTH a percentage AND a time reference matching our query window
        # Time window keywords: "since {date}", "from {date}", "compared to {date}"
        date_tokens = [source_date.strftime("%B %Y").lower(), source_date.strftime("%Y").lower()]
        has_time_ref = any(tok in combined for tok in date_tokens)

        if not has_time_ref:
            return None, f"no time reference matching {date_str} in search results"

        patterns = [
            # "steel prices rose 25% from June 2024 to..."
            r"(?:steel|aluminum|stainless)[^.]{0,80}?(?:rose|increased|up|gained)[^.]{0,40}?(\d+(?:\.\d+)?)\s*%",
            # "a 25% increase in steel prices since..."
            r"(\d+(?:\.\d+)?)\s*%\s*(?:increase|rise|gain)[^.]{0,60}?(?:steel|aluminum|stainless)",
            # "prices are 25% higher than..."
            r"(?:steel|aluminum|stainless)[^.]{0,60}?(?:are|is)\s*(\d+(?:\.\d+)?)\s*%\s*higher",
        ]
        for pat in patterns:
            m = re.search(pat, combined, re.IGNORECASE | re.DOTALL)
            if m:
                pct = float(m.group(1)) / 100.0
                if 0.0 < pct < 1.0:  # sanity: < 100% change
                    return pct, (answer or combined[:400])

        return None, f"no reliable percentage found. Snippet: {combined[:400]}"
    except Exception as e:
        return None, f"search error: {str(e)[:200]}"

def build_should_cost(g: dict) -> dict:
    part_name = g.get("part_name", "")
    part_description = st.session_state.get("raw_text", "")

    # Step 1: Call LLM (no price reference, no search)
    try:
        data = call_groq_should_cost(
            part_description, part_name, g["region"], g["material"],
            price_reference="",
        )
        validate_llm_rates(data)
    except Exception as e:
        st.session_state["llm_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        model = compute_should_cost_from_kb(g)
        model["source_mode"] = "KB fallback (LLM unavailable)"
        st.session_state["price_adjustment"] = None
        return model

    # Step 2: Check price source date. If >90 days, use KB directly.
    src_date_str = data.get("rm_price_source_date", "")
    src_date = parse_price_date(src_date_str)
    stale = is_stale(src_date)

    if stale:
        try:
            kb_rm = get_rm_row(g["material"], g["region"])
            kb_price = kb_rm["price_usd_per_kg"]
            llm_price = data["rm_price_usd_per_kg"]

            st.session_state["price_adjustment"] = {
                "reason": "LLM price source date is older than 90 days",
                "llm_price": llm_price,
                "llm_date": src_date_str or "unknown",
                "kb_price": kb_price,
                "kb_source": kb_rm["source"],
                "kb_effective": kb_rm["effective_date"],
            }

            # Overwrite LLM data with KB values so assemble_model_from_llm uses them
            data["rm_price_usd_per_kg"] = kb_price
            data["rm_source"] = f"KB benchmark ({kb_rm['source']}, eff. {kb_rm['effective_date']})"
            data["rm_currency"] = "USD"
            data["rm_unit_basis"] = "USD/kg"
            data["rm_price_source_date"] = kb_rm["effective_date"]

        except Exception:
            # If KB lookup fails, fall back to KB-computed model directly
            model = compute_should_cost_from_kb(g)
            model["source_mode"] = "KB fallback (LLM price stale, KB lookup failed)"
            st.session_state["price_adjustment"] = {
                "reason": "LLM price stale and KB lookup failed",
            }
            st.session_state["llm_error"] = None
            return model
    else:
        st.session_state["price_adjustment"] = None

    st.session_state["llm_error"] = None
    model = assemble_model_from_llm(data, g)
    model["source_mode"] = "LLM (Groq gpt-oss-120b)"
    if stale:
        model["source_mode"] += " + KB price override (LLM date stale)"
    return model

# ------------------------------------------------------------------
# FORECAST — 5 measurable drivers with real index growth + weights exposed
# ------------------------------------------------------------------
def forecast_weights(model: dict) -> dict:
    total = model["total"]
    rm_share = model["levers"]["Raw material"]["value"] / total
    conv_share = model["levers"]["Conversion (labor + machine)"]["value"] / total
    oh_share = model["levers"]["Overhead"]["value"] / total
    logi_share = model["levers"]["Logistics"]["value"] / total
    margin_share = model["levers"]["Margin"]["value"] / total

    weights = {
        "Raw Material": rm_share,
        "Labor": conv_share * 0.55,
        "Electricity": conv_share * 0.15 + 0.02,
        "Logistics": logi_share,
        "Overhead & Margin": oh_share + margin_share,
    }
    w_sum = sum(weights.values()) or 1
    return {k: v / w_sum for k, v in weights.items()}

def build_forecast(model: dict, months: int = 12) -> pd.DataFrame:
    start = datetime(2026, 10, 1)
    labels = [(start + pd.DateOffset(months=i)).strftime("%b-%Y") for i in range(months + 1)]

    total = model["total"]
    weights = forecast_weights(model)

    # Prefer LLM-supplied drivers when available
    if model.get("llm_forecast_drivers"):
        driver_map = {
            d["driver"]: d["annual_growth"]
            for d in model["llm_forecast_drivers"]
        }
    else:
        driver_map = {
            d: FORECAST_DRIVERS[d]["annual_growth"]
            for d in FORECAST_DRIVERS
        }

    rows = []
    for i, lbl in enumerate(labels):
        row = {"Month": lbl}
        composite_mult = 0.0
        for driver, w in weights.items():
            annual = driver_map.get(driver, 0.03)
            monthly = (1 + annual) ** (1 / 12) - 1
            factor = (1 + monthly) ** i
            row[driver] = round(total * w * factor, 4)
            composite_mult += w * factor
        row["Total"] = round(total * composite_mult, 4)
        rows.append(row)
    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# INPUT PARSING (user-friendly, no KB jargon)
# ------------------------------------------------------------------
def parse_user_description(text: str) -> dict:
    t = text.lower()
    out = {}

    for region in REGIONS:
        if region.lower() in t:
            out["region"] = region
            break
    if "region" not in out:
        out["region"] = "USA" if ("us" in t or "america" in t) else (REGIONS[0] if REGIONS else "USA")

    # --- Material matching: most specific first, avoid substring traps ---
    def pick(needle, exclude=()):
        """Find first grade containing needle but none of exclude."""
        for g in GRADES:
            gl = g.lower()
            if needle in gl and not any(x in gl for x in exclude):
                return g
        return None

    if "stainless" in t or re.search(r"\bss\b", t):
        out["material"] = pick("stainless") or GRADES[0]
    elif "alumin" in t:
        out["material"] = pick("alumin") or GRADES[0]
    elif "extru" in t:
        out["material"] = pick("extru") or GRADES[0]
    elif re.search(r"\b(a-?\d{2,4}|astm|aisi|a572|a36|1018|1045|q235|s355)\b", t) or "steel" in t:
        # Plain carbon / structural steel: exclude stainless and extrusion
        out["material"] = pick("steel", exclude=("stainless", "extru")) or GRADES[0]
    else:
        out["material"] = pick("steel", exclude=("stainless", "extru")) or GRADES[0]

    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:lbs?|pounds?)", t)
    if m:
        out["weight_kg"] = round(float(m.group(1)) * 0.453592, 3)
    else:
        m = re.search(r"(\d+(?:\.\d+)?)\s*kg", t)
        out["weight_kg"] = round(float(m.group(1)), 3) if m else 1.20

    m = re.search(r"(\d[\d,]*)\s*(?:pcs|pieces|units|nos)", t)
    out["volume"] = int(m.group(1).replace(",", "")) if m else 5000

    procs = []
    if any(k in t for k in ["laser", "cut", "blank", "shear"]):
        procs.append("cutting")
    if any(k in t for k in ["bend", "form", "press brake"]):
        procs.append("bending")
    if "weld" in t:
        procs.append("welding")
    if any(k in t for k in ["machin", "cnc", "mill", "drill"]):
        procs.append("machining")
    if any(k in t for k in ["powder coat", "paint", "galvani", "zinc", "anodiz", "finish"]):
        procs.append("surface_treatment")
    out["processes_applicable"] = procs or ["cutting", "bending"]

    out["industry"] = "General fabrication"
    for ind in KB["overhead"]["Industry"].unique():
        if ind.lower() in t:
            out["industry"] = ind
            break

    out["display_ccy"] = default_currency(out["region"])
    return out

# ------------------------------------------------------------------
# INPUT HELPERS — provenance + part-name based inference
# ------------------------------------------------------------------
BOUGHT_OUT_OPTIONS = [
    "None",
    "Fasteners / hardware (bolts, screws, rivets)",
    "Springs",
    "Bearings / bushings",
    "Inserts / threaded inserts",
    "Seals / gaskets / O-rings",
    "Clips / retainers",
    "Magnets",
    "Electrical connectors / terminals",
]

# Common part-name → process / material inference
PART_NAME_RULES = {
    "bracket": {"processes": ["cutting", "bending", "surface_treatment"]},
    "mount": {"processes": ["cutting", "bending", "surface_treatment"]},
    "enclosure": {"processes": ["cutting", "bending", "welding", "surface_treatment"]},
    "panel": {"processes": ["cutting", "bending", "surface_treatment"]},
    "cover": {"processes": ["cutting", "bending", "surface_treatment"]},
    "frame": {"processes": ["cutting", "bending", "welding"]},
    "chassis": {"processes": ["cutting", "bending", "welding", "surface_treatment"]},
    "housing": {"processes": ["cutting", "bending", "welding", "surface_treatment"]},
    "shaft": {"processes": ["machining"]},
    "pin": {"processes": ["machining"]},
    "bushing": {"processes": ["machining"]},
    "flange": {"processes": ["machining"]},
    "weldment": {"processes": ["cutting", "bending", "welding"]},
    "heat sink": {"processes": ["cutting", "machining", "surface_treatment"]},
    "heatsink": {"processes": ["cutting", "machining", "surface_treatment"]},
}


def infer_from_part_name(part_name: str) -> dict:
    """Infer processes/material from a part name when the description is sparse."""
    t = (part_name or "").lower()
    inferred = {"processes": [], "material": None, "notes": []}
    for key, rule in PART_NAME_RULES.items():
        if key in t:
            for p in rule["processes"]:
                if p not in inferred["processes"]:
                    inferred["processes"].append(p)
            inferred["notes"].append(f"Part name '{key}' → typical processes {rule['processes']}")
            break
    # Material cues in the part name
    if any(k in t for k in ["stainless", "ss-", "ss "]):
        inferred["material"] = "Stainless"
    elif "alum" in t:
        inferred["material"] = "Aluminum"
    elif any(k in t for k in ["steel", "bracket", "mount", "frame", "enclosure", "panel"]):
        inferred["material"] = "Steel Sheet"
    return inferred


def build_provenance(raw_text: str, ex: dict, part_name: str) -> dict:
    """Return a dict: field -> 'from description' | 'inferred from part name' | 'assumed default'."""
    t = (raw_text or "").lower()
    inferred = infer_from_part_name(part_name)
    prov = {}

    # Region
    if any(r.lower() in t for r in REGIONS):
        prov["Region"] = "from description"
    elif any(k in t for k in ["us", "usa", "america", "india", "china", "mexico", "europe", "germany"]):
        prov["Region"] = "from description"
    else:
        prov["Region"] = "assumed default"

    # Material
    if any(k in t for k in ["stainless", "alumin", "extru", "steel"]):
        prov["Material"] = "from description"
    elif inferred.get("material"):
        prov["Material"] = f"inferred from part name ({inferred['material']})"
    else:
        prov["Material"] = "assumed default"

    # Weight
    if re.search(r"(\d+(?:\.\d+)?)\s*(?:lbs?|pounds?|kg)", t):
        prov["Weight"] = "from description"
    else:
        prov["Weight"] = "assumed default"

    # Volume
    if re.search(r"(\d[\d,]*)\s*(?:pcs|pieces|units|nos)", t):
        prov["Volume"] = "from description"
    else:
        prov["Volume"] = "assumed default"

    # Processes
    if any(k in t for k in ["laser", "cut", "blank", "shear", "bend", "form", "weld",
                            "machin", "cnc", "mill", "drill", "powder coat", "paint",
                            "galvani", "zinc", "anodiz", "finish", "blast"]):
        prov["Processes"] = "from description"
    elif inferred.get("processes"):
        prov["Processes"] = "inferred from part name"
    else:
        prov["Processes"] = "assumed default"

    # Bought-out
    prov["Bought-out components"] = "from description" if any(
        k in t for k in ["bolt", "screw", "rivet", "spring", "bearing", "insert", "seal", "o-ring", "clip", "magnet", "connector"]
    ) else "assumed default"

    # Cut length / weld length / surface
    prov["Cut length"] = "from description" if re.search(r"\d+\s*mm", t) and "cut" in t else "assumed default"
    prov["Weld length"] = "from description" if "weld" in t and re.search(r"\d+\s*mm", t) else "assumed default"
    prov["Surface treatment"] = "from description" if any(
        k in t for k in ["powder", "paint", "galvani", "zinc", "anodiz", "blast", "finish"]
    ) else "assumed default"

    return prov

# ------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------
defaults = {
    "gathered": {},
    "model": None,
    "assumptions": [],
    "raw_text": "",
    "part_name": "",
    "step": 1,           # 1 = description, 2 = review extracted, 3 = model built
    "extracted": None,   # parsed dict from description
    "groq_rate_limited": False,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
st.sidebar.title("Should-Cost Agent")
pane = st.sidebar.radio("Navigate", ["Input Pane", "Should Cost Output", "Future Forecast", "Sources & Assumptions"])

if st.sidebar.button("Reset / New Analysis"):
    st.session_state["gathered"] = {}
    st.session_state["model"] = None
    st.session_state["assumptions"] = []
    st.session_state["raw_text"] = ""
    st.session_state["step"] = 1
    st.session_state["extracted"] = None
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("All rates cite their public source in the Sources pane.")

# ------------------------------------------------------------------
# PANE 1: INPUT (two-step)
# ------------------------------------------------------------------
if pane == "Input Pane":
    st.header("Input Pane")
    step = st.session_state["step"]

    # ---------- STEP 1: description + attachment only ----------
    if step == 1:
        st.caption(
            "Provide a part name (required) and optionally describe the part in your own words. "
            "The agent will extract region, material, weight, volume, and processes — and infer "
            "missing details from the part name where possible."
        )

        part_name = st.text_input(
            "Part name",
            value=st.session_state.get("part_name", ""),
            placeholder="e.g. Steel mounting bracket, Stainless enclosure, Aluminum heat sink",
        )

        raw = st.text_area(
            "Part description (optional)",
            value=st.session_state["raw_text"],
            height=140,
            placeholder="e.g. Steel bracket, 1.2 kg, laser cut and 6 bends, 5000 pcs/year, made in USA. Powder coat finish.",
        )

        uploaded_file = st.file_uploader(
            "Optional attachment (drawing, image, or BOM)",
            type=["pdf", "png", "jpg", "jpeg", "csv", "xlsx"],
        )
        if uploaded_file is not None:
            st.caption(
                f"Attached: **{uploaded_file.name}** ({uploaded_file.size/1024:.1f} KB). "
                "The agent will use the description as the primary source and can cross-reference the attachment."
            )

        # Live enable — recomputed every rerun as the user types
        can_proceed = bool(part_name.strip()) or bool(raw.strip())

        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Next → Review extracted values", type="primary", disabled=not can_proceed):
                st.session_state["raw_text"] = raw
                st.session_state["part_name"] = part_name
                st.session_state["extracted"] = parse_user_description(raw or part_name)
                # augment with part-name inference for anything missing
                inferred = infer_from_part_name(part_name)
                if inferred.get("processes") and not st.session_state["extracted"].get("processes_applicable"):
                    st.session_state["extracted"]["processes_applicable"] = inferred["processes"]
                st.session_state["step"] = 2
                st.rerun()
        with col_b:
            if st.button("Skip description and enter manually"):
                st.session_state["extracted"] = {
                    "region": REGIONS[0],
                    "material": GRADES[0],
                    "industry": KB["overhead"]["Industry"].iloc[0],
                    "weight_kg": 1.20,
                    "volume": 5000,
                    "processes_applicable": ["cutting", "bending"],
                    "display_ccy": default_currency(REGIONS[0]),
                }
                st.session_state["raw_text"] = ""
                st.session_state["part_name"] = part_name
                st.session_state["step"] = 2
                st.rerun()

    # ---------- STEP 2: review & adjust extracted values ----------
    elif step == 2:
        ex = st.session_state["extracted"] or {}
        prov = build_provenance(
            st.session_state["raw_text"],
            ex,
            st.session_state.get("part_name", ""),
        )
        st.caption(
            "Review and adjust the extracted values. Each field shows whether it was captured "
            "from your description, inferred from the part name, or assumed as a default."
        )

        with st.expander("Original description", expanded=False):
            st.write(st.session_state["raw_text"] or "_(none — manual entry)_")

        # --- Region / Material / Industry ---
        c1, c2, c3 = st.columns(3)
        with c1:
            region_sel = st.selectbox(
                "Region", REGIONS,
                index=REGIONS.index(ex.get("region")) if ex.get("region") in REGIONS else 0,
            )
            st.caption(f"_{prov['Region']}_")
        with c2:
            grade_sel = st.selectbox(
                "Material", GRADES,
                index=GRADES.index(ex.get("material")) if ex.get("material") in GRADES else 0,
                help="Stainless / Aluminum / Extrusion / Steel sheet grades.",
            )
            st.caption(f"_{prov['Material']}_")
        with c3:
            industries = KB["overhead"]["Industry"].unique().tolist()
            ind_sel = st.selectbox(
                "Industry", industries,
                index=industries.index(ex.get("industry")) if ex.get("industry") in industries else 0,
            )
            st.caption("_assumed default_")

        # --- Weight / Volume / Currency ---
        c4, c5, c6 = st.columns(3)
        with c4:
            weight = st.number_input("Weight (kg)", min_value=0.01, max_value=500.0,
                                     value=float(ex.get("weight_kg", 1.20)), step=0.05)
            st.caption(f"_{prov['Weight']}_")
        with c5:
            volume = st.number_input("Annual volume (pcs)", min_value=1, max_value=10_000_000,
                                     value=int(ex.get("volume", 5000)), step=100)
            st.caption(f"_{prov['Volume']}_")
        with c6:
            ccy_options = ["USD", "INR", "CNY", "EUR", "MXN", "JPY"]
            default_ccy = default_currency(region_sel)
            ccy = st.selectbox("Display currency", ccy_options,
                               index=ccy_options.index(default_ccy) if default_ccy in ccy_options else 0)
            st.caption("_assumed default_")

        st.markdown("---")
        st.markdown("**Manufacturing inputs**")
        st.caption(f"_{prov['Processes']}_ — the agent selects the operations below based on your description and part name.")

        # --- Physical operation inputs (replaces raw process checkboxes) ---
        op1, op2, op3 = st.columns(3)
        with op1:
            cut_len = st.number_input(
                "Cut length (mm) — laser cut / blanking",
                min_value=0, max_value=50_000, value=int(ex.get("cut_perimeter_mm", 1800)), step=50,
            )
            st.caption(f"_{prov['Cut length']}_")
        with op2:
            weld_len = st.number_input(
                "Weld length (mm)",
                min_value=0, max_value=50_000, value=int(ex.get("weld_length_mm", 420)), step=50,
            )
            st.caption(f"_{prov['Weld length']}_")
        with op3:
            surface_options = ["None", "Powder coating", "Wet paint", "Shot blasting", "Zinc plating", "Anodizing"]

            # Infer the default from the parsed processes + raw text
            def _default_surface(ex_procs, raw_text):
                t = (raw_text or "").lower()
                if "powder" in t:
                    return "Powder coating"
                if "wet paint" in t or ("paint" in t and "powder" not in t):
                    return "Wet paint"
                if "blast" in t:
                    return "Shot blasting"
                if "zinc" in t or "galvani" in t:
                    return "Zinc plating"
                if "anodiz" in t:
                    return "Anodizing"
                if "surface_treatment" in ex_procs:
                    return "Powder coating"  # safe default when parser said surface treatment but no specific keyword
                return "None"

            _default_surf = _default_surface(ex.get("processes_applicable", []), st.session_state.get("raw_text", ""))
            surface_sel = st.selectbox(
                "Surface treatment", surface_options,
                index=surface_options.index(_default_surf),
            )
            st.caption(f"_{prov['Surface treatment']}_")

        op4, op5 = st.columns(2)
        with op4:
            n_bends = st.number_input(
                "Number of bends / formed features (hydraulic press)",
                min_value=0, max_value=200, value=int(ex.get("n_bends", 6)), step=1,
            )
        with op5:
            bought_out = st.selectbox(
                "Bought-out components",
                BOUGHT_OUT_OPTIONS,
                index=0,
            )
            st.caption(f"_{prov['Bought-out components']}_")

        # Derive processes from physical inputs (do not show raw process list to user)
        selected_procs = []
        if cut_len > 0:
            selected_procs.append("cutting")
        if n_bends > 0:
            selected_procs.append("bending")
        if weld_len > 0:
            selected_procs.append("welding")
        if surface_sel != "None":
            selected_procs.append("surface_treatment")
        # Part-name inferred machining (shafts/pins/flanges) still honoured
        inferred = infer_from_part_name(st.session_state.get("part_name", ""))
        if "machining" in inferred.get("processes", []) and "machining" not in selected_procs:
            selected_procs.append("machining")
        if not selected_procs:
            selected_procs = ["cutting"]

        col_back, col_build = st.columns([1, 2])
        with col_back:
            if st.button("← Back to description"):
                st.session_state["step"] = 1
                st.rerun()
        with col_build:
            if st.button("Build Should Cost", type="primary"):
                g = {
                    "region": region_sel, "material": grade_sel, "industry": ind_sel,
                    "weight_kg": weight, "volume": volume,
                    "processes_applicable": selected_procs,
                    "cut_perimeter_mm": cut_len,
                    "weld_length_mm": weld_len,
                    "n_bends": n_bends,
                    "surface_area_m2": float(ex.get("surface_area_m2", 0.18)),
                    "bought_out": bought_out,
                    "display_ccy": ccy,
                    "part_name": st.session_state.get("part_name", ""),
                }
                st.session_state["gathered"] = g
                st.session_state["model"] = build_should_cost(g)
                st.session_state["assumptions"] = [
                    f"Part name = {g['part_name'] or '(none)'}",
                    f"Region = {region_sel}  ({prov['Region']})",
                    f"Material = {material_family(grade_sel)}  ({prov['Material']})",
                    f"Industry = {ind_sel}",
                    f"Weight = {weight} kg  ({prov['Weight']})",
                    f"Volume = {volume:,} pcs/yr  ({prov['Volume']})",
                    f"Cut length = {cut_len} mm  ({prov['Cut length']})",
                    f"Weld length = {weld_len} mm  ({prov['Weld length']})",
                    f"Bends / formed features = {n_bends}",
                    f"Surface treatment = {surface_sel}  ({prov['Surface treatment']})",
                    f"Bought-out = {bought_out}  ({prov['Bought-out components']})",
                ]
                st.session_state["step"] = 3
                st.success("Should-cost built. Open Should Cost Output from the sidebar.")
                st.rerun()

    # ---------- STEP 3: done ----------
    else:
        # Neutral, non-technical success message — same regardless of source
        st.success("Should cost has been built. Open **Should Cost Output** or **Future Forecast** from the sidebar.")

        # ---- Model source notice (collapsed, plain English) ----
        model = st.session_state.get("model") or {}
        source_mode = model.get("source_mode", "")
        llm_err = st.session_state.get("llm_error")
        rate_limited = bool(llm_err and ("429" in llm_err or "rate_limit" in llm_err.lower()))

        if llm_err:
            if rate_limited:
                short_reason = (
                    "The AI service reached its daily usage limit for this moment. "
                    "To keep your analysis moving, the should-cost was generated using the "
                    "curated public rate-card library (Knowledge Base) instead."
                )
            else:
                short_reason = (
                    "The AI service was temporarily unavailable. "
                    "The should-cost was generated using the curated public rate-card "
                    "library (Knowledge Base) instead."
                )
            with st.expander("ℹ️ How this should-cost was generated", expanded=False):
                st.write(short_reason)
                st.caption(
                    "Both methods use the same cost model and formula. "
                    "The AI method applies real-time reasoning over published indices; "
                    "the Knowledge-Base method uses curated benchmark rates. "
                    "Numbers may differ slightly between the two."
                )
                with st.expander("Technical detail (for reviewers)", expanded=False):
                    st.code(llm_err, language="text")
        else:
            with st.expander("ℹ️ How this should-cost was generated", expanded=False):
                st.write(
                    "The should-cost was generated using AI reasoning over your part "
                    "description and current published market indices."
                )

        # ---- Live price reference (unchanged, already collapsed) ----
        price_ref = st.session_state.get("price_search_result")
        if price_ref:
            with st.expander("Live price reference used (from browser search)", expanded=False):
                st.caption(price_ref)

        adj = st.session_state.get("price_adjustment")
        if adj:
            with st.expander("ℹ️ Raw material price source", expanded=False):
                if "error" in adj:
                    st.caption(adj["error"])
                elif "reason" in adj:
                    st.caption(
                        f"{adj['reason']}. "
                        f"LLM returned **${adj['llm_price']:.2f}/kg** (as of {adj['llm_date']}), "
                        f"which is stale. Using KB benchmark **${adj['kb_price']:.2f}/kg** "
                        f"from *{adj['kb_source']}* (effective {adj['kb_effective']})."
                    )
                else:
                    st.caption(
                        f"Adjusted from **${adj['original']:.2f}/kg** to "
                        f"**${adj['adjusted']:.2f}/kg** ({adj['pct']:+.1%})."
                    )

        price_err = st.session_state.get("price_search_error")
        if price_err:
            with st.expander("ℹ️ Market price lookup", expanded=False):
                st.caption(
                    "A live web-price check could not be completed for this run. "
                    "The should-cost uses the curated rate-card price for this material "
                    "and region."
                )
                with st.expander("Technical detail (for reviewers)", expanded=False):
                    st.code(price_err, language="text")

        # ---- Navigation buttons (unchanged) ----
        if st.button("Edit inputs"):
            st.session_state["step"] = 2
            st.rerun()
        if st.button("Start a new analysis"):
            st.session_state["step"] = 1
            st.session_state["gathered"] = {}
            st.session_state["model"] = None
            st.session_state["assumptions"] = []
            st.session_state["raw_text"] = ""
            st.session_state["extracted"] = None
            st.session_state["part_name"] = ""
            st.session_state["llm_error"] = None
            st.session_state["price_search_result"] = None
            st.session_state["price_search_error"] = None
            st.session_state["groq_rate_limited"] = False
            st.rerun()

# ------------------------------------------------------------------
# PANE 2: SHOULD COST OUTPUT
# ------------------------------------------------------------------
elif pane == "Should Cost Output":
    st.header("Should Cost Output")
    model = st.session_state.get("model")
    if not model:
        st.info("Build a model in the Input Pane first.")
    else:
        total = model["total"]
        disp_ccy = model["display_ccy"]
        conv = fx_rate("USD", disp_ccy)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Should cost / part", f"{total*conv:,.2f} {disp_ccy}")
        k2.metric("Should cost / part (USD)", f"${total:,.2f}")
        k3.metric("Annual volume", f"{model['volume']:,} pcs")
        k4.metric("Annual spend", f"${total*model['volume']:,.0f}")

        # --- 6 high-level cost levers summary table ---
        st.markdown("#### High-level cost levers")
        lever_rows = [
            ("Raw Material",              model["levers"]["Raw material"]["value"]),
            ("Manufacturing Processes",   model["levers"]["Conversion (labor + machine)"]["value"]),
            ("Overhead (SG&A, Factory, R&D)",
                                          model["levers"]["Overhead"]["value"] + model["levers"]["Electricity"]["value"]),
            ("Packaging & Forwarding (Logistics)",
                                          model["levers"]["Logistics"]["value"]),
            ("Supplier Margin",           model["levers"]["Margin"]["value"]),
            ("Tool Cost (if applicable)", 0.0),  # not modelled in current engine
        ]
        summary = pd.DataFrame([
            {"Cost lever": n, "USD per unit": f"${v:,.2f}", "Share of total": f"{v/total:.1%}" if total else "0.0%"}
            for n, v in lever_rows
        ] + [{"Cost lever": "TOTAL COST PER UNIT", "USD per unit": f"${total:,.2f}", "Share of total": "100.0%"}])
        st.dataframe(summary, use_container_width=True, hide_index=True)

        # --- Chart ---
        st.markdown("#### Cost lever breakdown")
        names = list(model["levers"].keys())
        vals = [model["levers"][n]["value"] for n in names]
        fig, ax = plt.subplots(figsize=(9, 3.5))
        bars = ax.barh(names[::-1], vals[::-1], color="#4C78A8")
        ax.set_xlabel("USD per part")
        ax.set_title("Cost lever contribution (USD per part)")
        for i, (bar, v) in enumerate(zip(bars, vals[::-1])):
            ax.text(v + max(vals)*0.02, i, f" ${v:,.2f}", va="center", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("#### Lever detail")
        for name in names:
            lev = model["levers"][name]
            pct = lev["value"] / total * 100 if total else 0
            with st.expander(f"{name} — ${lev['value']:,.2f}  ({pct:.1f}%)"):
                for k, v in lev["detail"].items():
                    if k == "lines" and isinstance(v, list):
                        st.dataframe(pd.DataFrame(v), use_container_width=True, hide_index=True)
                    else:
                        st.write(f"**{k.replace('_',' ').title()}:** {v}")

        st.markdown("#### Cost narrative")
        st.markdown(build_narrative(model))

# ------------------------------------------------------------------
# PANE 3: FUTURE FORECAST
# ------------------------------------------------------------------
elif pane == "Future Forecast":
    st.header("Future Forecast — rolling 12 months")
    model = st.session_state.get("model")
    if not model:
        st.info("Build a model in the Input Pane first.")
    else:
        total = model["total"]
        fc = build_forecast(model, months=12)
        disp_ccy = model.get("display_ccy", "USD")
        conv = fx_rate("USD", disp_ccy)
        grade_family = material_family(model["material"])

        st.caption(
            "Each driver grows at its published index rate (steel/aluminium PPI, ECI manufacturing wages, "
            "EIA industrial electricity, freight PPI) weighted by its share of current should-cost."
        )

        start_total = float(fc["Total"].iloc[0])
        end_total = float(fc["Total"].iloc[-1])
        delta_pct = (end_total / start_total - 1.0) * 100 if start_total else 0.0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Current (Oct-2026)", f"{start_total*conv:,.2f} {disp_ccy}")
        k2.metric("12-month outlook (Oct-2027)", f"{end_total*conv:,.2f} {disp_ccy}")
        k3.metric("Projected change", f"{delta_pct:+.1f}%")
        k4.metric("Annual volume", f"{model['volume']:,} pcs")

        # ---- (A) Part Unit Cost movement (primary chart) ----
        st.markdown("#### Part unit cost movement (USD per part)")
        fig, ax = plt.subplots(figsize=(11, 4.0))
        x = np.arange(len(fc))
        y_total = fc["Total"].values.astype(float)
        ax.plot(x, y_total, color="#1f4e79", linewidth=2.4,
                marker="o", markersize=4, label="Total should-cost")
        ax.fill_between(x, y_total, color="#1f4e79", alpha=0.10)

        # Data labels — two decimals
        for xi, yi in zip(x, y_total):
            ax.annotate(f"${yi:,.2f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7, color="#1f4e79")

        # Monthly X-axis (every month, smaller font)
        ax.set_xticks(x)
        ax.set_xticklabels(fc["Month"], rotation=60, ha="right", fontsize=7)

        # Small headroom so labels don't clip
        pad = (y_total.max() - y_total.min()) * 0.15 if y_total.max() > y_total.min() else y_total.max() * 0.05
        ax.set_ylim(y_total.min() - pad, y_total.max() + pad)

        ax.set_ylabel("USD per part", fontsize=9)
        ax.set_title("Rolling 12-month total should-cost per unit", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ---- (B) Separate small charts for relevant cost levers (native units) ----
        st.markdown("#### Cost lever trajectories")

        # Determine the native unit & rate for each lever from the KB-backed model detail
        rm_detail = model["levers"]["Raw material"]["detail"]
        conv_detail = model["levers"]["Conversion (labor + machine)"]["detail"]
        elec_detail = model["levers"]["Electricity"]["detail"]
        logi_detail = model["levers"]["Logistics"]["detail"]

        # Raw material — use the KB's own unit basis (USD/kg, USD/tonne, USD/lb)
        rm_unit_basis = str(rm_detail["unit_basis"]).lower()
        if "tonne" in rm_unit_basis or "/t" in rm_unit_basis or "ton" in rm_unit_basis:
            rm_factor, rm_unit_label = 1000.0, "USD / tonne"
        elif "lb" in rm_unit_basis:
            rm_factor, rm_unit_label = 0.453592, "USD / lb"
        else:
            rm_factor, rm_unit_label = 1.0, "USD / kg"

        # Labor — USD per hour (rate is already USD/hr in the model)
        labor_unit_label = "USD / hour"
        labor_factor = 1.0

        # Electricity — USD per kWh
        elec_unit_label = "USD / kWh"
        elec_factor = 1.0

        # Logistics — USD per kg (freight rate per kg)
        logi_unit_label = "USD / kg"
        logi_factor = 1.0

        raw_material_label = material_family(model["material"])  # e.g. "Stainless Steel"

        # Each entry: (driver_key, chart_title, native_factor, native_unit_label, colour, source_value)
        lever_charts = [
            ("Raw Material", raw_material_label,
             rm_factor, rm_unit_label, "#c0504d",
             rm_detail["price_usd_per_kg"]),

            ("Labor", "Labor",
             labor_factor, labor_unit_label, "#4f81bd",
             conv_detail["labor_usd_per_hr"]),

            ("Electricity", "Electricity",
             elec_factor, elec_unit_label, "#e6a817",
             elec_detail["rate_usd_per_kwh"]),

            ("Logistics", "Freight & Logistics",
             logi_factor, logi_unit_label, "#54a24b",
             logi_detail["freight_usd_per_kg"]),
        ]

        grid = st.columns(2)
        for idx, (driver_key, label, factor, unit_label, colour, base_rate) in enumerate(lever_charts):
            with grid[idx % 2]:
                fig, ax = plt.subplots(figsize=(5.4, 2.9))

                # Driver trajectory = share-of-total × Total / quantity-basis
                # Convert the USD-per-part driver series into the lever's native unit:
                # native_value(t) = base_rate × (driver_series(t) / driver_series(t0))
                # because the driver series already reflects growth applied to that lever.
                driver_series = fc[driver_key].values.astype(float) if driver_key in fc else np.zeros(len(fc))
                base_share = driver_series[0] if driver_series[0] else 1.0
                native_series = base_rate * (driver_series / base_share) * factor

                ax.plot(x, native_series, color=colour, linewidth=2.0, marker="o", markersize=3)
                ax.fill_between(x, native_series, color=colour, alpha=0.15)

                # End-point label for readability
                ax.annotate(f"{native_series[-1]:,.2f} {unit_label}",
                            (x[-1], native_series[-1]), textcoords="offset points",
                            xytext=(-6, 8), ha="right", fontsize=7, color=colour)

                ax.set_title(label, fontsize=10)
                # Monthly X-axis (every month)
                ax.set_xticks(x)
                ax.set_xticklabels(fc["Month"], rotation=60, ha="right", fontsize=6.5)
                ax.set_ylabel(unit_label, fontsize=8)
                ax.tick_params(axis="y", labelsize=8)
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # ---- Forecast table ----
        st.markdown("#### Forecast table")
        table = fc.copy()
        for c in table.columns:
            if c != "Month":
                table[c] = table[c].map(lambda v: f"${v:,.2f}")
        st.dataframe(table, use_container_width=True, hide_index=True)

        # ---- Driver weights + index reference ----
        st.markdown("#### Forecast drivers and index references")
        weights = forecast_weights(model)
        driver_rows = [{
            "Driver": d,
            "Weight in total": f"{weights.get(d, 0.0):.1%}",
            "Annual growth assumed": f"{FORECAST_DRIVERS[d]['annual_growth']:.1%}",
            "Public index": FORECAST_DRIVERS[d]["index"],
            "Note": FORECAST_DRIVERS[d]["note"],
        } for d in FORECAST_DRIVERS]
        st.dataframe(pd.DataFrame(driver_rows), use_container_width=True, hide_index=True)

        # ---- Annualized spend outlook ----
        st.markdown("#### Annualized spend outlook")
        spend = fc[["Month", "Total"]].copy()
        spend["Annual spend (USD)"] = spend["Total"] * model["volume"]
        spend["Total"] = spend["Total"].map(lambda v: f"${v:,.2f}")
        spend["Annual spend (USD)"] = spend["Annual spend (USD)"].map(lambda v: f"${v:,.0f}")
        st.dataframe(spend, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------
# PANE 4: SOURCES & ASSUMPTIONS
# ------------------------------------------------------------------
elif pane == "Sources & Assumptions":
    st.header("Sources & Assumptions")
    model = st.session_state.get("model")
    if not model:
        st.info("Build a model in the Input Pane first. Sources and assumptions are shown after a model is built.")
        st.markdown("#### Forecast index references (always available)")
        ref_rows = [{
            "Driver": d,
            "Public index": FORECAST_DRIVERS[d]["index"],
            "Annual growth assumed": f"{FORECAST_DRIVERS[d]['annual_growth']:.1%}",
            "Note": FORECAST_DRIVERS[d]["note"],
        } for d in FORECAST_DRIVERS]
        st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)
    else:
        st.markdown("#### Input assumptions")
        if st.session_state.get("assumptions"):
            for a in st.session_state["assumptions"]:
                st.write(f"- {a}")
        else:
            st.write("_No assumptions recorded._")

        st.markdown("#### Rate sources used in this model")
        src_rows = []

        rm = model["levers"]["Raw material"]["detail"]
        src_rows.append({
            "Cost element": "Raw material",
            "Rate used": f"${rm['price_usd_per_kg']:,.2f}/kg "
                         f"(from {rm['price_local']:,.2f} {rm['currency']} {rm['unit_basis']})",
            "Source": rm["source"],
            "Effective date": rm["effective_date"],
        })

        conv = model["levers"]["Conversion (labor + machine)"]["detail"]
        src_rows.append({
            "Cost element": "Labor",
            "Rate used": f"${conv['labor_usd_per_hr']:,.2f}/hr",
            "Source": conv["labor_source"],
            "Effective date": "—",
        })
        for line in conv["lines"]:
            src_rows.append({
                "Cost element": f"Machine — {line['process']}",
                "Rate used": f"${line['machine_usd_hr']:,.2f}/hr",
                "Source": line["source"],
                "Effective date": "—",
            })

        elec = model["levers"]["Electricity"]["detail"]
        src_rows.append({
            "Cost element": "Electricity",
            "Rate used": f"${elec['rate_usd_per_kwh']:,.4f}/kWh",
            "Source": elec["source"],
            "Effective date": "2025",
        })

        oh = model["levers"]["Overhead"]["detail"]
        src_rows.append({
            "Cost element": "Overhead",
            "Rate used": f"{oh['pct']:.1%} of (RM + conversion + energy)",
            "Source": oh["source"],
            "Effective date": "—",
        })

        logi = model["levers"]["Logistics"]["detail"]
        src_rows.append({
            "Cost element": "Logistics",
            "Rate used": f"${logi['freight_usd_per_kg']:,.2f}/kg freight + "
                         f"${logi['packaging_usd']:,.2f} packaging",
            "Source": logi["source"],
            "Effective date": "—",
        })

        mg = model["levers"]["Margin"]["detail"]
        src_rows.append({
            "Cost element": "Margin",
            "Rate used": f"{mg['pct']:.1%} of pre-margin cost",
            "Source": mg["source"],
            "Effective date": "—",
        })

        st.dataframe(pd.DataFrame(src_rows), use_container_width=True, hide_index=True)

        st.markdown("#### FX rates applied")
        st.caption(
            f"Display currency: **{model['display_ccy']}**. "
            f"USD→{model['display_ccy']} = {fx_rate('USD', model['display_ccy']):.4f}. "
            "The underlying cost model is computed in USD and converted to the display currency "
            "at the published rate shown here."
        )

        st.markdown("#### Method notes")
        st.markdown(
            "- **Raw material** = gross weight (weight × (1 + scrap %)) × USD/kg rate for the selected grade and region.\n"
            "- **Conversion** = sum of machine-hour costs for each operation at the regional machine rate "
            "(bending and surface treatment use the hydraulic-press rate).\n"
            "- **Electricity** = estimated kWh (scaled from conversion time) × EIA industrial average rate.\n"
            "- **Overhead** = regional/industry overhead % applied to (RM + conversion + energy).\n"
            "- **Logistics** = part weight × regional freight $/kg + regional packaging cost.\n"
            "- **Margin** = regional/industry margin % applied to pre-margin subtotal (subtotal + logistics).\n"
            "- **Forecast** = each driver grown at its published index rate, weighted by its current share of total, "
            "compounded monthly over 12 months."
        )

        st.markdown("---")
        st.caption(
            "This is an indicative should-cost model for procurement negotiation support. "
            "Validate supplier-specific process times, scrap rates, and freight lanes before relying on the result."
        )
