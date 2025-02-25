import argparse
import csv
import json
import os
import time

import requests


def main(args):

    # Request to get number of total pokemon
    all_pokemon = fetch_json(
        "https://pokeapi.co/api/v2/pokemon?limit=100000&offset=0", delay=args.delay
    )
    num_pkmn = all_pokemon["count"]

    # run through pokemon and grab data
    metadata, type_set, egg_group_set, color_set, shape_set = request_data(
        num_pkmn=num_pkmn, delay=args.delay
    )

    # save down metadata to file
    metadata_path = os.path.join(args.output_dir, "metadata.json")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    type_path = os.path.join(args.output_dir, "types.csv")

    with open(type_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(type_set)

    egg_path = os.path.join(args.output_dir, "egg_groups.csv")

    with open(egg_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(egg_group_set)

    color_path = os.path.join(args.output_dir, "colors.csv")

    with open(color_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(color_set)

    shape_path = os.path.join(args.output_dir, "shapes.csv")

    with open(shape_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(shape_set)


def fetch_json(url, delay=0.1):
    """Fetch JSON data from a URL with error handling and rate limiting."""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an error for 4XX/5XX responses
        time.sleep(delay)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None  # Return None instead of crashing


def request_data(num_pkmn, delay=0.1):
    # initialize storage variables for metadata
    metadata = {}
    type_set = set()
    egg_group_set = set()
    color_set = set()
    shape_set = set()

    for i in range(1, num_pkmn + 1):
        base_data = fetch_json(f"https://pokeapi.co/api/v2/pokemon/{i}/", delay=delay)
        if not base_data:
            continue
        types = []
        for type in base_data["types"]:
            types.append(type["type"]["name"])
            type_set.add(type["type"]["name"])

        species_data = fetch_json(
            f"https://pokeapi.co/api/v2/pokemon-species/{i}/", delay=delay
        )
        if not species_data:
            continue
        egg_groups = []
        for egg_group in species_data["egg_groups"]:
            egg_groups.append(egg_group["name"])
            egg_group_set.add(egg_group["name"])
        color_set.add(species_data["color"]["name"])
        shape_set.add(species_data["shape"]["name"])
        metadata[str(i)] = {
            "types": types,
            "egg_groups": egg_groups,
            "color": species_data["color"]["name"],
            "shape": species_data["shape"]["name"],
        }
        time.sleep(delay)

    return metadata, type_set, egg_group_set, color_set, shape_set


def save_csv(data_set, output_dir, filename):
    """Save a set to a CSV file."""
    path = os.path.join(output_dir, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/metadata",
        help="Directory to store metadata",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Delay between API calls",
    )
    args = parser.parse_args()

    main(args)
