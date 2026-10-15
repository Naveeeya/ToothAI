"""Tooth notation: FDI (ISO 3950) <-> Universal, and tooth names. Permanent dentition only."""

# Universal 1-16 run from the patient's upper right third molar (FDI 18) to upper left (28),
# 17-32 from lower left third molar (38) to lower right (48).
_UNIVERSAL_ORDER = (
    [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
    + [38, 37, 36, 35, 34, 33, 32, 31, 41, 42, 43, 44, 45, 46, 47, 48]
)
FDI_TO_UNIVERSAL = {fdi: i + 1 for i, fdi in enumerate(_UNIVERSAL_ORDER)}
UNIVERSAL_TO_FDI = {u: fdi for fdi, u in FDI_TO_UNIVERSAL.items()}

QUADRANT_NAMES = {1: "Upper right", 2: "Upper left", 3: "Lower left", 4: "Lower right"}
_POSITION_NAMES = {
    1: "central incisor", 2: "lateral incisor", 3: "canine", 4: "first premolar",
    5: "second premolar", 6: "first molar", 7: "second molar", 8: "third molar",
}
TOOTH_TYPES = {1: "incisor", 2: "incisor", 3: "canine", 4: "premolar", 5: "premolar",
               6: "molar", 7: "molar", 8: "molar"}


def is_valid_fdi(fdi):
    return fdi in FDI_TO_UNIVERSAL


def tooth_name(fdi):
    """'Lower left first molar' for 36."""
    q, p = divmod(int(fdi), 10)
    return f"{QUADRANT_NAMES[q]} {_POSITION_NAMES[p]}"


def tooth_type(fdi):
    return TOOTH_TYPES[int(fdi) % 10]


def label(fdi, notation="FDI"):
    """Display label for a tooth in the chosen notation ('FDI' or 'Universal')."""
    if notation == "Universal":
        return str(FDI_TO_UNIVERSAL[int(fdi)])
    return str(int(fdi))
