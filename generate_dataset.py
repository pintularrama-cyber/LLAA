import pandas as pd
import random
from datetime import datetime, timedelta

def generate_unique_lla():
    # Listas de componentes para crear combinaciones únicas
    projects = [
        ("Prime Combined Cycle", "Canada", "Non-EU"), ("North Nuclear Refurb", "France", "EU"),
        ("East Solar PV", "Germany", "EU"), ("Global CCGT", "Japan", "Non-EU"),
        ("Delta Nuclear", "Saudi Arabia", "Non-EU"), ("South Wind Farm", "Poland", "EU"),
        ("Mega Biomass", "Mexico", "Non-EU"), ("Alpha Wind Project", "Australia", "Non-EU"),
        ("Prime Refinery", "Brazil", "Non-EU"), ("West Hydrogen Plant", "France", "EU"),
        ("East LNG Terminal", "Netherlands", "EU"), ("Central Refinery", "Italy", "EU"),
        ("East Wind Farm", "Belgium", "EU"), ("South Hydro-power", "Sweden", "EU"),
        ("Mega Solar PV", "USA", "Non-EU"), ("Delta LNG", "UAE", "Non-EU")
    ]

    categories = [
        "WELDING & NDT", "PAINTING & COATING", "PRESERVATION", 
        "DOCUMENTATION", "CE MARKING / PED", "FLANGE MANAGEMENT", 
        "VENDOR QUALITY", "SITE INSTALLATION"
    ]

    # Problemas técnicos variados
    problems = [
        {"cat": "WELDING & NDT", "title": "Incorrect interpass temperature on P91", "desc": "Welders failed to maintain the 250-300C range leading to hardness out of specs.", "act": "Install digital thermal sensors with auto-logging."},
        {"cat": "PAINTING & COATING", "title": "Osmotic blistering in C5-M environment", "desc": "Surface was contaminated with chlorides (>20mg/m2) before primer application.", "act": "Mandatory salt test (Bresle method) before every coat."},
        {"cat": "PRESERVATION", "title": "Dew point sensor failure in turbine", "desc": "Relative humidity exceeded 40% during shipment due to faulty desiccant.", "act": "Dual redundant sensors with GPS tracking for transit."},
        {"cat": "DOCUMENTATION", "title": "MTC 3.1 Traceability loss", "desc": "Heat numbers on pipes were painted over, losing link to the digital dossier.", "act": "Use hard-stamping or RFID tags for all high-pressure piping."},
        {"cat": "CE MARKING / PED", "title": "Wrong Category for Pressure Vessel", "desc": "Equipment arrived marked as Cat II instead of Cat III according to PED 2014/68/EU.", "act": "Cross-check design calcs with NoBo before manufacturing starts."},
        {"cat": "FLANGE MANAGEMENT", "title": "Stress corrosion on SS316 bolts", "desc": "Zinc-plated nuts were used on stainless steel flanges causing contamination.", "act": "Strict segregation of carbon and stainless steel fasteners in warehouse."},
        {"cat": "VENDOR QUALITY", "title": "Counterfeit valves detected", "desc": "Internal trim material did not match the specification despite having certificates.", "act": "Implement Positive Material Identification (PMI) at vendor shop."},
        {"cat": "CE MARKING / PED", "title": "ATEX certificate missing for Zone 1 pump", "desc": "Motor had ATEX but the pump coupling did not, voiding the assembly rating.", "act": "Mandatory 'Ignition Hazard Assessment' for all coupled assemblies."},
        {"cat": "SITE INSTALLATION", "title": "Foundation anchor bolt misalignment", "desc": "The template shifted during concreting, exceeding the 2mm tolerance.", "act": "Use laser scanning to verify bolt positions before concrete sets."}
    ]

    data = []
    start_date = datetime(2018, 1, 1)

    for i in range(600):
        ref = 5500 + i
        proj, loc, region = random.choice(projects)
        prob = random.choice(problems)
        
        # Si no es Europa, evitamos que la categoría sea CE MARKING para que el filtro tenga sentido
        final_cat = prob["cat"]
        if region == "Non-EU" and "CE MARKING" in final_cat:
            final_cat = "DOCUMENTATION"
            
        gen_date = (start_date + timedelta(days=random.randint(0, 2000))).strftime("%d/%m/%Y")
        
        # Añadimos ruido/variación técnica para que no sean idénticos
        tech_detail = random.choice(["Area A", "Unit 100", "Package 05", "System 20", "Substation B"])
        
        data.append({
            "LL Ref": ref,
            "Title": f"QA/QC: {prob['title']} - {tech_detail}",
            "Status": "Published",
            "Department": "090 - CALIDAD",
            "Project": f"{proj} ({loc})",
            "Knowledge Category": final_cat,
            "Description": f"In {proj} ({loc}), we identified: {prob['desc']} (Location: {tech_detail}).",
            "Action Proposed": prob["act"],
            "Public Comments / Implementation Plan": "Standard implementation for all new projects. Follow-up during internal audits.",
            "Generation Date": gen_date
        })

    return pd.DataFrame(data)

# Generar y guardar
df_new = generate_unique_lla()
df_new.to_csv("lecciones_aprendidas_calidad_600_v2.csv", sep=";", index=False)
print("¡Archivo generado con 600 lecciones únicas!")