# Logistics_Data_Analysis
# Logistics Data Preprocessing

## Project Objective
The objective of this project is to collect, clean, and preprocess logistics and supply chain data to prepare it for further analysis. This helps improve data quality and supports better decision-making in logistics operations.

## Dataset
The project uses a supply chain logistics dataset containing information about delivery times, shipping costs, fuel consumption, warehouse inventory, traffic congestion, supplier reliability, and other logistics factors.

## Tools Used
- Python
- Pandas
- NumPy
- Visual Studio Code

## Data Preprocessing Steps
1. Loaded the logistics dataset using Pandas.
2. Checked for missing values and duplicate records.
3. Converted relevant columns into appropriate data types.
4. Handled missing numerical values using median imputation when required.
5. Identified potential outliers using the Interquartile Range (IQR) method.
6. Scaled selected numerical features using Min-Max scaling.
7. Exported the processed dataset and outlier summary as CSV files.

## Project Files
— Python script for data preprocessing.
 — Input dataset.
 — Processed output dataset (generated when the script runs).
 — Summary of detected outliers.
 — Project report.

## How to Run the Project

### 1. Install Python libraries
```bash
pip install pandas numpy
```

### 2. Keep the files together
Place `preprocessing.py` and `dynamic_supply_chain_logistics_dataset.csv` in the same folder.

### 3. Run the script
```bash
python preprocessing.py
```

The script should generate the processed dataset and outlier summary in the output location specified in the Python code.

## Expected Outcome
The project produces a structured dataset ready for further logistics analysis, along with an outlier summary to help understand unusual values.

## Conclusion
This project demonstrates basic data cleaning, preprocessing, outlier detection, and numerical feature scaling using Python for logistics and supply chain analysis.
