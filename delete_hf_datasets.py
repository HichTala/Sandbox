from huggingface_hub import HfApi

api = HfApi()

# Replace with your actual dataset repository names
datasets_to_delete = [
    "HichTala/deepfruits_10",
    "HichTala/cadot_10shot_1",
    "HichTala/cadot_5shot_1"
]

shots = [5, 10, 1]
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
datasets = ["uodd", "deepfruits", "cadot", "fashionpedia", "xview", "dior", "dota","oktoberfest", "artaxor"]

for shot in shots:
    for seed in seeds:
        for dataset in datasets:
            try:
                api.delete_repo(repo_id=f"HichTala/{dataset}_{shot}shot_{seed}", repo_type="dataset")
                print(f"Successfully deleted: HichTala/{dataset}_{shot}shot_{seed}")
            except Exception as e:
                print(f"Failed to delete HichTala/{dataset}_{shot}shot_{seed}: {e}")