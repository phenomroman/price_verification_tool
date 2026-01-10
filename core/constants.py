import os
import json

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the models directory
MODELS_DIR = os.path.join(BASE_DIR, "price_models")

# Define path to the JSON config file
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

# Load configuration at module level
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Features
YEAR_FEATURES = config["YEAR_FEATURES"]
NUM_FEATURES = config["NUM_FEATURES"]
CAT_FEATURES = config["CAT_FEATURES"]
ALL_FEATURES = YEAR_FEATURES + NUM_FEATURES + CAT_FEATURES

# Goods Information
GOODS_INFO = config["GOODS_INFO"]

# Country Options
COUNTRY_OPTIONS = config["COUNTRY_OPTIONS"]

# Port Options
PORT_OPTIONS = config["PORT_OPTIONS"]

# Currency Options
CURRENCY_OPTIONS = config["CURRENCY_OPTIONS"]

# Incoterm Options
INCOTERM_OPTIONS = config["INCOTERM_OPTIONS"]

# Goods with insufficient training data
INSUFFICIENT_DATA_CODES = ['58071000', '96061000']

