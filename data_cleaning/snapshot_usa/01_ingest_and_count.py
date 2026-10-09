"""Stage 1: read the Snapshot USA deployments and annotated sequences, and count detections of each
species in each deployment.

Inputs (datasets/raw/snapshotUSA_2019-2023/):
  ssusa_finaldeployments.csv  one row per camera deployment: location, start/end date, survey nights, habitat
  ssusa_finalsequences.csv    one row per annotated detection sequence (a burst of images): taxon, group size

Join key: (Year, Deployment_ID). Deployment_ID alone is NOT unique: 46 IDs are reused in a later year
for the same site (a join on the ID alone puts 4,710 sequences outside their deployment's dates).
With the key all sequences fall inside the survey dates; 69 sequences have no deployment row and are dropped.

Counts per (deployment, species), species = Genus + Species (non-empty `Species`):
  n_sequences  annotated sequences of the species
  n_events     independent events: sequences of the species closer than EVENT_GAP_MIN minutes to the
               end of the previous one (at the same deployment) are merged, so a deer that lingers in front of
               the camera counts once. This is what the encounter rate uses.
Records not identified to species (Bird, Mammal, Rodent, 'Peromyscus Species', 'Sciuridae Family', 'Animal',
'Unknown', ...) are counted per deployment in `n_seq_unresolved` (absence of a species is less certain where
this is high); vehicle records are ignored.

Writes (gitignored, regenerable) datasets/raw/snapshotUSA_2019-2023/derived/:
  deployments_clean.parquet   one row per deployment, with coordinate decimals and date fields
  species_counts.parquet      one row per (deployment, species) with at least one sequence
  taxa.parquet                species table: scientific name, common name(s), class, human/domestic flag
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "datasets/raw/snapshotUSA_2019-2023"
OUT = RAW / "derived"
EVENT_GAP_MIN = 30
# humans and domestic / livestock species: kept in the table with `is_human_or_domestic` = True, not scored
HUMAN_DOMESTIC = {
    "Homo sapiens", "Bos taurus", "Canis familiaris", "Felis catus", "Equus caballus", "Equus asinus",
    "Capra aegagrus hircus", "Ovis aries",
}
VEHICLE = r"Vehicle|ATV|Motorcycle"


def n_decimals(s):
    """Decimals in the source text of a coordinate, ignoring trailing zeros."""
    return s.fillna("").str.split(".").str[1].fillna("").str.rstrip("0").str.len()


def main():
    OUT.mkdir(exist_ok=True)
    d = pd.read_csv(RAW / "ssusa_finaldeployments.csv", dtype={"Latitude": str, "Longitude": str})
    d["n_decimals"] = np.minimum(n_decimals(d.Latitude), n_decimals(d.Longitude))
    d["Latitude"] = pd.to_numeric(d.Latitude, errors="coerce")
    d["Longitude"] = pd.to_numeric(d.Longitude, errors="coerce")
    d["Start_Date"] = pd.to_datetime(d.Start_Date)
    d["End_Date"] = pd.to_datetime(d.End_Date)
    assert not d.duplicated(["Year", "Deployment_ID"]).any()
    d.to_parquet(OUT / "deployments_clean.parquet", index=False)

    s = pd.read_csv(RAW / "ssusa_finalsequences.csv", usecols=["Year", "Deployment_ID", "Start_Time", "End_Time", "Class",
                                                                "Genus", "Species", "Common_Name"])
    n0 = len(s)
    s = s.merge(d[["Year", "Deployment_ID"]], on=["Year", "Deployment_ID"], how="inner")
    print(f"sequences: {n0}; with a deployment row: {len(s)} ({n0 - len(s)} dropped)")
    s["Start_Time"] = pd.to_datetime(s.Start_Time)
    s["End_Time"] = pd.to_datetime(s.End_Time)
    for c in ["Genus", "Species", "Common_Name", "Class"]:
        s[c] = s[c].str.strip()

    sp = s[s.Species != ""].copy()
    sp["species"] = sp.Genus + " " + sp.Species
    un = s[(s.Species == "") & ~s.Common_Name.str.contains(VEHICLE)]
    unresolved = un.groupby(["Year", "Deployment_ID"]).size().rename("n_seq_unresolved").reset_index()

    # independent events: new event when the sequence starts more than EVENT_GAP_MIN after the latest end so far
    sp = sp.sort_values(["Year", "Deployment_ID", "species", "Start_Time"]).reset_index(drop=True)
    key = ["Year", "Deployment_ID", "species"]
    prev_end = sp.groupby(key).End_Time.transform(lambda x: x.cummax().shift())
    new = prev_end.isna() | ((sp.Start_Time - prev_end) > pd.Timedelta(minutes=EVENT_GAP_MIN))
    sp["event"] = new
    counts = sp.groupby(key).agg(n_sequences=("Start_Time", "size"), n_events=("event", "sum")).reset_index()
    counts["n_events"] = counts.n_events.astype(int)
    counts.to_parquet(OUT / "species_counts.parquet", index=False)
    unresolved.to_parquet(OUT / "unresolved_counts.parquet", index=False)

    taxa = sp.groupby("species").agg(common_name=("Common_Name", lambda v: " | ".join(sorted(set(v)))),
                                     taxon_class=("Class", "first"), n_sequences=("Start_Time", "size")).reset_index()
    taxa["is_human_or_domestic"] = taxa.species.isin(HUMAN_DOMESTIC)
    taxa.to_parquet(OUT / "taxa.parquet", index=False)

    print(f"deployments: {len(d)}; species: {len(taxa)}; (deployment, species) rows: {len(counts)}")
    print(f"sequences of species: {len(sp)} -> independent events: {int(counts.n_events.sum())} ({EVENT_GAP_MIN}-min rule)")
    print(f"unresolved (not to species) sequences: {len(un)} in {len(unresolved)} deployments")
    print("human/domestic species present:", taxa[taxa.is_human_or_domestic].species.tolist())


if __name__ == "__main__":
    main()
