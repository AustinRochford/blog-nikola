# /// script
# dependencies = [
#     "nflreadpy",
#     "polars",
#     "requests",
# ]
# ///
"""Build a CSV of NFL stadium locations and elevations.

Geocodes every distinct `game_stadium` value in the play-by-play data via
Nominatim, looks up elevation via Open-Elevation, and produces two groupings
per row:

- `canonical_venue`: same physical *site* (used for the elevation join /
  post-hoc correlation check against the stadium random effect).
- `building`: same physical *building* (used as the random effect's grouping
  key itself) - splits apart cases where `canonical_venue` merges two
  different, demolished-and-rebuilt buildings on the same site.

Bare-name geocoding produces confidently wrong matches for names that
collide with unrelated places (e.g. "Nissan Stadium" -> Yokohama, Japan;
"Giants Stadium" -> Indianapolis). KNOWN_BAD_ALIASES and MANUAL_HINTS below
were built by manually cross-checking every result against known NFL
stadium history - don't trust a re-run's geocode results without doing the
same.
"""

import time

import nflreadpy as nfl
import polars as pl
import requests

OUTPUT_PATH = "stadiums.csv"

HEADERS = {"User-Agent": "fg-irt-blog-research/1.0 (elevation lookup for stadium random effect validation)"}

# disambiguation hints for names that are ambiguous or international
GEOCODE_HINTS = {
    "Allianz Arena": "Allianz Arena, Munich, Germany",
    "Arena Corinthians": "Arena Corinthians, Sao Paulo, Brazil",
    "Azteca Stadium": "Estadio Azteca, Mexico City, Mexico",
    "Deutsche Bank Park": "Deutsche Bank Park, Frankfurt, Germany",
    "Rogers Centre": "Rogers Centre, Toronto, Canada",
    "Tottenham Stadium": "Tottenham Hotspur Stadium, London, UK",
    "Twickenham Stadium": "Twickenham Stadium, London, UK",
    "Wembley Stadium": "Wembley Stadium, London, UK",
    "The Coliseum": "Oakland Coliseum, Oakland, CA, USA",
    "Memorial Stadium (Champaign)": "Memorial Stadium, Champaign, Illinois, USA",
    "Tiger Stadium (LSU)": "Tiger Stadium, Baton Rouge, Louisiana, USA",
    "Husky Stadium": "Husky Stadium, Seattle, Washington, USA",
    "Sun Devil Stadium": "Sun Devil Stadium, Tempe, Arizona, USA",
}

# aliases whose own geocode is verified wrong (name collision with an unrelated
# place) - never trust these, always fall through to another alias or a
# CANONICAL_HINTS override
KNOWN_BAD_ALIASES = {
    "Candlestick Park", "Dolphin Stadium", "Giants Stadium", "Nissan Stadium",
    "Georgia Dome", "Edward Jones Dome", "O.co Coliseum", "Texas Stadium",
    "Veterans Stadium", "Monster Park",
}

# canonical_venue (same physical site) -> aliases sharing that site
CANONICAL_GROUPS = {
    "Candlestick Park (SF)": ["Candlestick Park", "3Com Park", "Monster Park"],
    "MetLife Stadium site (East Rutherford, NJ)": ["MetLife Stadium", "Giants Stadium", "New Meadowlands Stadium"],
    "Dome at America's Center (St. Louis)": ["Dome at America's Center", "Edward Jones Dome", "TWA Dome"],
    "Atlanta dome site": ["Mercedes-Benz Stadium", "Georgia Dome"],
    "Louisiana Superdome (New Orleans)": ["Louisiana Superdome", "Mercedes-Benz Superdome"],
    "Oakland Coliseum": [
        "Oakland-Alameda County Coliseum", "The Coliseum", "McAfee Coliseum",
        "Network Associates Coliseum", "O.co Coliseum", "Ring Central Coliseum",
    ],
    "Hard Rock Stadium (Miami Gardens)": ["Hard Rock Stadium", "Pro Player Stadium", "Dolphin Stadium", "Sun Life Stadium"],
    "Nissan Stadium (Nashville)": ["LP Field", "Adelphia Coliseum", "Nissan Stadium"],
    "Jacksonville stadium": ["Jacksonville Municipal Stadium", "Alltel Stadium", "EverBank Field", "TIAA Bank Stadium"],
    "Paycor Stadium (Cincinnati)": ["Paycor Stadium", "Paul Brown Stadium"],
    "Acrisure Stadium (Pittsburgh)": ["Acrisure Stadium", "Heinz Field", "Three Rivers Stadium"],
    "Cleveland stadium": ["Cleveland Browns Stadium", "FirstEnergy Stadium"],
    "Bank of America Stadium (Charlotte)": ["Bank of America Stadium", "Ericsson Stadium"],
    "Buffalo stadium": ["New Era Field", "Ralph Wilson Stadium"],
    "M&T Bank Stadium (Baltimore)": ["M&T Bank Stadium", "Ravens Stadium", "PSINet Stadium"],
    "FedExField (Landover, MD)": ["FedExField", "Jack Kent Cooke Stadium"],
    "Mile High site (Denver)": [
        "Empower Field at Mile High", "Invesco Field at Mile High",
        "Mile High Stadium", "Sports Authority Field at Mile High",
    ],
    "U.S. Bank Stadium (Minneapolis)": ["U.S. Bank Stadium", "Mall of America Field", "Hubert H. Humphrey Metrodome"],
    "NRG Stadium (Houston)": ["NRG Stadium", "Reliant Stadium"],
    "Lumen Field (Seattle)": ["Lumen Field", "CenturyLink Field", "Qwest Field", "Seahawks Stadium", "Seattle Kingdome"],
    "Arrowhead Stadium (Kansas City)": ["Arrowhead Stadium", "GEHA Field at Arrowhead Stadium"],
    "AT&T Stadium (Arlington, TX)": ["AT&T Stadium", "Cowboys Stadium"],
    "Lincoln Financial Field (Philadelphia)": ["Lincoln Financial Field", "Veterans Stadium"],
    "Gillette Stadium (Foxborough)": ["Gillette Stadium", "Foxboro Stadium"],
    "State Farm Stadium (Glendale, AZ)": ["State Farm Stadium", "University of Phoenix Stadium"],
    "Lucas Oil Stadium (Indianapolis)": ["Lucas Oil Stadium", "RCA Dome"],
    "Texas Stadium (Irving, TX)": ["Texas Stadium"],
    "Alamo Dome (San Antonio)": ["Alamo Dome"],
    "Qualcomm Stadium (San Diego)": ["Qualcomm Stadium"],
    "Cinergy Field (Cincinnati)": ["Cinergy Field"],
}

# manual, well-documented queries for aliases with no usable sibling geocode
CANONICAL_HINTS = {
    "Candlestick Park (SF)": "Candlestick Point, San Francisco, California, USA",
    "Dome at America's Center (St. Louis)": "America's Center, St. Louis, Missouri, USA",
    "Buffalo stadium": "Highmark Stadium, Orchard Park, New York, USA",
    "FedExField (Landover, MD)": "Northwest Stadium, Landover, Maryland, USA",
    "Texas Stadium (Irving, TX)": "Irving, Texas, USA",
    "Alamo Dome (San Antonio)": "Alamodome, San Antonio, Texas, USA",
    "Cinergy Field (Cincinnati)": "Great American Ball Park, Cincinnati, Ohio, USA",  # same riverfront site
}

# canonical_venue -> {building: [aliases]}, for groups that span >1 physical
# building (demolished-and-rebuilt on the same site); everything else keeps
# its canonical_venue as a single building
BUILDING_SPLITS = {
    "MetLife Stadium site (East Rutherford, NJ)": {
        "New Meadowlands Stadium": ["MetLife Stadium", "New Meadowlands Stadium"],
        "Giants Stadium": ["Giants Stadium"],
    },
    "Atlanta dome site": {
        "Mercedes-Benz Stadium": ["Mercedes-Benz Stadium"],
        "Georgia Dome": ["Georgia Dome"],
    },
    "Acrisure Stadium (Pittsburgh)": {
        "Heinz Field": ["Acrisure Stadium", "Heinz Field"],
        "Three Rivers Stadium": ["Three Rivers Stadium"],
    },
    "Mile High site (Denver)": {
        "Invesco Field at Mile High": [
            "Empower Field at Mile High", "Invesco Field at Mile High",
            "Sports Authority Field at Mile High",
        ],
        "Mile High Stadium": ["Mile High Stadium"],
    },
    "U.S. Bank Stadium (Minneapolis)": {
        "U.S. Bank Stadium": ["U.S. Bank Stadium"],
        "Hubert H. Humphrey Metrodome": ["Mall of America Field", "Hubert H. Humphrey Metrodome"],
    },
    "Lumen Field (Seattle)": {
        "Seahawks Stadium": ["Lumen Field", "CenturyLink Field", "Qwest Field", "Seahawks Stadium"],
        "Seattle Kingdome": ["Seattle Kingdome"],
    },
    "Lincoln Financial Field (Philadelphia)": {
        "Lincoln Financial Field": ["Lincoln Financial Field"],
        "Veterans Stadium": ["Veterans Stadium"],
    },
    "Gillette Stadium (Foxborough)": {
        "Gillette Stadium": ["Gillette Stadium"],
        "Foxboro Stadium": ["Foxboro Stadium"],
    },
    "Lucas Oil Stadium (Indianapolis)": {
        "Lucas Oil Stadium": ["Lucas Oil Stadium"],
        "RCA Dome": ["RCA Dome"],
    },
}

# canonical_venue key -> the first name the structure actually went by,
# for non-split groups where the alias list's first entry isn't the first
# chronological name (only groups that need overriding are listed here;
# groups not listed default to their first alias, which is correct for them)
BUILDING_FIRST_NAME = {
    "Dome at America's Center (St. Louis)": "TWA Dome",
    "Hard Rock Stadium (Miami Gardens)": "Pro Player Stadium",
    "Nissan Stadium (Nashville)": "Adelphia Coliseum",
    "Paycor Stadium (Cincinnati)": "Paul Brown Stadium",
    "Bank of America Stadium (Charlotte)": "Ericsson Stadium",
    "Buffalo stadium": "Ralph Wilson Stadium",
    "M&T Bank Stadium (Baltimore)": "Ravens Stadium",
    "FedExField (Landover, MD)": "Jack Kent Cooke Stadium",
    "AT&T Stadium (Arlington, TX)": "Cowboys Stadium",
    "State Farm Stadium (Glendale, AZ)": "University of Phoenix Stadium",
    "NRG Stadium (Houston)": "Reliant Stadium",
}


def geocode(query):
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": query, "format": "json", "limit": 1},
        headers=HEADERS,
        timeout=10,
    )
    data = resp.json()
    if data:
        return float(data[0]["lat"]), float(data[0]["lon"])
    return None, None


def geocode_all_aliases(names):
    results = {}
    for name in names:
        query = GEOCODE_HINTS.get(name, name)
        lat, lon = geocode(query)
        time.sleep(1.1)
        if lat is None and name not in GEOCODE_HINTS:
            lat, lon = geocode(f"{name}, USA")
            time.sleep(1.1)
        results[name] = None if name in KNOWN_BAD_ALIASES else (lat, lon)
    return results


def resolve_canonical_coords(alias_coords):
    coords = {}
    for canonical, aliases in CANONICAL_GROUPS.items():
        lat = lon = None
        for a in aliases:
            r = alias_coords.get(a)
            if r is not None:
                lat, lon = r
                break
        if lat is None:
            lat, lon = geocode(CANONICAL_HINTS[canonical])
            time.sleep(1.1)
        coords[canonical] = (lat, lon)
    return coords


def fetch_elevations(canonical_coords):
    names = list(canonical_coords)
    locations = [{"latitude": lat, "longitude": lon} for lat, lon in canonical_coords.values()]
    resp = requests.post(
        "https://api.open-elevation.com/api/v1/lookup",
        json={"locations": locations},
        timeout=60,
    )
    resp.raise_for_status()
    elevations_m = [r["elevation"] for r in resp.json()["results"]]
    return dict(zip(names, elevations_m))


def build_building_map():
    building_of = {}
    for canonical, aliases in CANONICAL_GROUPS.items():
        if canonical in BUILDING_SPLITS:
            for building, split_aliases in BUILDING_SPLITS[canonical].items():
                first_name = BUILDING_FIRST_NAME.get(building, building)
                for a in split_aliases:
                    building_of[a] = first_name
        else:
            first_name = BUILDING_FIRST_NAME.get(canonical, aliases[0])
            for a in aliases:
                building_of[a] = first_name
    return building_of


def main():
    pbp = nfl.load_pbp(seasons=True)
    stadium_names = (
        pbp.select("game_stadium").unique().drop_nulls().sort("game_stadium")["game_stadium"].to_list()
    )

    grouped = {a for aliases in CANONICAL_GROUPS.values() for a in aliases}
    for name in stadium_names:
        if name not in grouped:
            CANONICAL_GROUPS[name] = [name]

    alias_coords = geocode_all_aliases(stadium_names)
    canonical_coords = resolve_canonical_coords(alias_coords)
    elevations_m = fetch_elevations(canonical_coords)
    building_of = build_building_map()

    rows = []
    for canonical, aliases in CANONICAL_GROUPS.items():
        lat, lon = canonical_coords[canonical]
        elev_m = elevations_m[canonical]
        for alias in aliases:
            rows.append({
                "game_stadium": alias,
                "building": building_of[alias],
                "canonical_venue": canonical,
                "lat": lat,
                "lon": lon,
                "elevation_m": round(elev_m, 1),
                "elevation_ft": round(elev_m * 3.28084, 1),
            })

    df = pl.DataFrame(rows).sort("elevation_ft", descending=True)
    df.write_csv(OUTPUT_PATH)
    print(f"wrote {df.shape[0]} rows, {df['building'].n_unique()} buildings, {df['canonical_venue'].n_unique()} venues")


if __name__ == "__main__":
    main()
