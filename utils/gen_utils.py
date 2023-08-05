import re

# create pattern to find column headers
key_pattern = re.compile("`(.*?) = (.*?)`")

# create pattern to find feature mapping
value_pattern = re.compile("(\d+) : (.*)")

# create dictionary for renaming features and column headers
features_dict = {}
header_dict = {}

def extract_mapping(data):
    # splitting the input into separate sections
    sections = re.split("\$", data)

    for section in sections:
        key_match = re.search(key_pattern, section)
        if key_match:
            # saving key match
            key_german = key_match.group(1)
            key_english = key_match.group(2)
            # finding the value pairs
            value_matches = re.findall(value_pattern, section)
            # creating the value dictionary
            value_dict = {int(k): v.strip() for k, v in value_matches}
            # adding the key-value pair to the output dictionary
            features_dict[key_english] = value_dict
            # adding mapping to header dictionary
            header_dict[key_german] = key_english
    return features_dict, header_dict