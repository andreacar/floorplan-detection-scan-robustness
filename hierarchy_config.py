"""
Hierarchy configuration for CubiCasaVec → RT-DETR.

Final taxonomy = 17 classes.
Fireplace is merged into FURNITURE_OTHER.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ============================================================
# Level-2 final detection classes
# ============================================================

'''LEVEL2_CLASSES = [
    # Structural
    "WALL",
    "COLUMN",
    "STAIR",
    "RAILING",

    # Openings
    "DOOR",
    "WINDOW",

    # Spaces
    #"SPACE_LIVING",
    #"SPACE_BED",
    #"SPACE_KITCHEN",
    #"SPACE_BATH",
    #"SPACE_CIRC",
    #"SPACE_OUTDOOR",
    #"SPACE_OTHER",

    # Furniture
    #"TOILET",
    #"BATH_SHOWER",
    #"KITCHEN_UNIT",
    #"FURNITURE_OTHER",
]'''

LEVEL2_CLASSES = [
    "WALL",
    "COLUMN",
    "STAIR",
    "RAILING",
    "DOOR",
    "WINDOW",
]


LEVEL2_NAME_TO_IDX = {c: i for i, c in enumerate(LEVEL2_CLASSES)}

# ============================================================
# Raw → Final Level-2 mapping
# (Uses substring or exact matching as appropriate)
# ============================================================

RAW_TO_L2 = {}

# -----------------------------
# Structural
# -----------------------------
for k in [
    "Wall", "Wall ", "Wall External"
]:
    RAW_TO_L2[k] = "WALL"

for k in [
    "Column Circle", "Column FreeShape", "Column Rectangle"
]:
    RAW_TO_L2[k] = "COLUMN"

RAW_TO_L2["Flight"] = "STAIR"
RAW_TO_L2["Railing"] = "RAILING"

# -----------------------------
# Openings
# -----------------------------
DOOR_KEYS = [
    "Door None Beside",
    "Door ParallelSlide Beside",
    "Door RollUp Beside",
    "Door Slide Beside",
    "Door Swing Beside",
    "Door Swing Opposite",
    "Door Zfold Beside",
    "Doors",
    "Door Fold Beside",
]
for k in DOOR_KEYS:
    RAW_TO_L2[k] = "DOOR"

WINDOW_KEYS = [
    "Window Regular",
    "Glass",
    "Window Full",
    "Window Sauna",
]
for k in WINDOW_KEYS:
    RAW_TO_L2[k] = "WINDOW"

# -----------------------------
# Spaces
# -----------------------------
SPACE_LIVING = [
    "Space LivingRoom",
    "Space Dining",
    "Space Den Fireplace",
    "Space Room",
    "Space RecreationRoom",
]
for k in SPACE_LIVING:
    RAW_TO_L2[k] = "SPACE_LIVING"

SPACE_BED = [
    "Space Bedroom",
    "Space Bedroom Guest",
    "Space Alcove",
    "Space Closet WalkIn",
]
for k in SPACE_BED:
    RAW_TO_L2[k] = "SPACE_BED"

SPACE_KITCHEN = [
    "Space Kitchen",
    "Space Kitchen Kitchenette",
    "Space Kitchen Open",
]
for k in SPACE_KITCHEN:
    RAW_TO_L2[k] = "SPACE_KITCHEN"

SPACE_BATH = [
    "Space Bath",
    "Space Bath Shower",
    "Space Sauna",
    "Space Utility Laundry",
    "Space TechnicalRoom Boiler",
]
for k in SPACE_BATH:
    RAW_TO_L2[k] = "SPACE_BATH"

SPACE_CIRC = [
    "Space Hall",
    "Space Entry",
    "Space Entry Lobby",
    "Space DraughtLobby",
    "Space DressingRoom",
]
for k in SPACE_CIRC:
    RAW_TO_L2[k] = "SPACE_CIRC"

SPACE_OUTDOOR = [
    "Space Outdoor",
    "Space Outdoor Balcony",
    "Space Outdoor Balcony Glazed",
    "Space Outdoor CoveredArea",
    "Space Outdoor Porch",
    "Space Outdoor Terrace",
    # Any patio/veranda variants:
    "Space Outdoor Balcony Covered",
    "Space Outdoor Patio",
    "Space Outdoor Patio Glazed",
    "Space Outdoor Veranda",
    "Space Outdoor Veranda Glazed",
]
for k in SPACE_OUTDOOR:
    RAW_TO_L2[k] = "SPACE_OUTDOOR"

# Everything else that is a space
OTHER_SPACES = [
    "Space Library",
    "Space RetailSpace",
    "Space Storage",
    "Space Storage Cold",
    "Space Storage Fuel",
    "Space Storage Oil",
    "Space Storage Wood",
    "Space Storage Shed",
    "Space Garage",
    "Space Office",
    "Space TechnicalRoom",
    "Space UserDefined",
    "Space Undefined",
    "Space Basement",
    "Space SwimmingPool",
    "Space Library Archive",
    "Space ExerciseRoom Gym",
    "Space OpenToBelow",
    "Space CarPort",
]
for k in OTHER_SPACES:
    RAW_TO_L2[k] = "SPACE_OTHER"

# -----------------------------
# Furniture groups
# -----------------------------

TOILET_KEYS = ["FixedFurniture Toilet", "FixedFurniture Urinal"]
for k in TOILET_KEYS:
    RAW_TO_L2[k] = "TOILET"

BATH_SHOWER_KEYS = [
    "FixedFurniture Bathtub",
    "FixedFurniture BathtubRound",
    "FixedFurniture Shower",
    "FixedFurniture ShowerCab",
    "FixedFurniture ShowerScreen",
    "FixedFurniture ShowerPlatform",
]
for k in BATH_SHOWER_KEYS:
    RAW_TO_L2[k] = "BATH_SHOWER"

KITCHEN_UNIT_KEYS = [
    "FixedFurniture BaseCabinet",
    "FixedFurniture WallCabinet",
    "FixedFurniture ElectricalAppliance IntegratedStove",
    "FixedFurniture ElectricalAppliance Dishwasher",
    "FixedFurniture ElectricalAppliance Refrigerator",
    "FixedFurniture ElectricalAppliance WashingMachine",
    "FixedFurniture ElectricalAppliance TumbleDryer",
    "FixedFurniture ElectricalAppliance SpaceForAppliance",
    "FixedFurniture ElectricalAppliance SpaceForAppliance2",
]
for k in KITCHEN_UNIT_KEYS:
    RAW_TO_L2[k] = "KITCHEN_UNIT"

# EVERYTHING ELSE → FURNITURE_OTHER
def map_raw_to_l2(name: str):
    """
    Return None for anything we do NOT want to train on.
    Dataset code must skip None labels.
    """
    name = name.strip()

    if name in RAW_TO_L2:
        mapped = RAW_TO_L2[name]
        if mapped in LEVEL2_CLASSES:
            return mapped

    # Explicitly ignore everything else
    return None



# ============================================================
# API used by dataset_hier.py
# ============================================================

def load_level2_classes_and_mapping(json_path=None):
    """json_path is ignored (compatibility only)."""
    return LEVEL2_CLASSES, RAW_TO_L2

def build_level2_to_level1_map(level2_classes):
    """Dummy L1 mapping (optional)."""
    return {c: c for c in level2_classes}

# Level-1/0 not used, but must exist for compatibility:
LEVEL1_TO_IDX = {c: i for i, c in enumerate(LEVEL2_CLASSES)}
LEVEL1_TO_LEVEL0 = {c: c for c in LEVEL2_CLASSES}
LEVEL0_TO_IDX = {c: i for i, c in enumerate(LEVEL2_CLASSES)}
