import os
import json


def read_json_files(directory):
    results = []

    # Iterate over all files and subdirectories in the given directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)

                # Read the contents of the .json file
                with open(file_path, "r") as json_file:
                    data = json.load(json_file)
                    results.append(data)

    return results


directory_path = "results_"
json_data = read_json_files(directory_path)
for model_result in json_data:
    print("Model: ", model_result['existing_model'])
    print("Test ROC: ", model_result['test_roc'])
    print("Test AP: ", model_result['test_ap'])


# Process the json_data as needed

