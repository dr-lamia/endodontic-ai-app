from __future__ import annotations


def proposed_management(pulpal: str, apical: str, systemic_signs: int, tooth_unrestorable: int) -> str:
    """Transparent research decision-support pathway, not an autonomous prescription."""
    if tooth_unrestorable:
        return (
            "Assess restorability and strategic value. If the tooth is confirmed unrestorable, "
            "extraction may be indicated; manage acute infection/pain as clinically required."
        )

    if apical == "Acute apical abscess":
        plan = (
            "Urgent endodontic assessment. Establish drainage when indicated/feasible and provide "
            "definitive source control (root canal treatment or extraction according to restorability)."
        )
        if systemic_signs:
            plan += (
                " Systemic involvement is present: assess urgently for spreading infection and "
                "consider systemic antimicrobial therapy only when clinically indicated according to local guidance."
            )
        else:
            plan += (
                " Routine systemic antibiotics are not automatically indicated in the absence of "
                "systemic/spreading infection."
            )
        return plan

    if pulpal == "Normal pulp" and apical == "Normal apical tissues":
        return (
            "No endodontic treatment indicated from these findings; investigate and manage "
            "the source of symptoms if present."
        )

    if pulpal == "Reversible pulpitis":
        return (
            "Conservative management of the irritant/caries with preservation of pulp vitality where feasible; "
            "select restorative or vital-pulp approach according to exposure status and full clinical assessment, then review."
        )

    if pulpal in {"Symptomatic irreversible pulpitis", "Asymptomatic irreversible pulpitis"}:
        return (
            "Definitive pulpal treatment is indicated. Consider an evidence-based vital pulp treatment or root canal treatment "
            "according to pulp exposure, bleeding/hemostasis findings, tooth/restorative factors, patient factors, and clinician assessment."
        )

    if pulpal == "Pulp necrosis":
        if apical == "Chronic apical abscess":
            return (
                "Definitive source control is indicated: nonsurgical root canal treatment when the tooth is restorable, "
                "with drainage as appropriate; extraction if not maintainable."
            )
        return (
            "Nonsurgical root canal treatment is generally indicated for a restorable tooth with necrotic pulp, "
            "with management of apical disease and follow-up; extraction is an alternative when the tooth cannot be maintained."
        )

    if apical in {
        "Symptomatic apical periodontitis",
        "Asymptomatic apical periodontitis",
        "Chronic apical abscess",
    }:
        return (
            "Identify and treat the endodontic source according to pulpal status and restorability; "
            "endodontic treatment and follow-up are commonly required when disease is of endodontic origin."
        )

    return "Complete clinical and radiographic assessment is required before a definitive management decision."
