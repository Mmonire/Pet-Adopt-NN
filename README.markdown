# Pet Adoption Prediction Project

## Overview
This project implements a deep neural network using **Keras** to predict whether a pet will be adopted based on a dataset of pet characteristics. The goal is to preprocess the data, build and train a neural network, and generate predictions for a test dataset. The project includes data preprocessing steps, model construction, training, and evaluation, with the final output formatted for submission.

The dataset contains information about pets, such as their type, age, breed, and health status, with the target variable indicating whether the pet was adopted within a certain time frame. The project is implemented in a Jupyter Notebook (`new_home.ipynb`) and uses Python libraries like **pandas**, **numpy**, **scikit-learn**, **keras**, and **category_encoders**.

## Project Structure
The repository contains the following files:
- **new_home.ipynb**: The main Jupyter Notebook containing the complete workflow, including data loading, preprocessing, model building, training, and prediction.
- **submission.csv**: The output file containing predictions for the test dataset.
- **test.csv**: The preprocessed test dataset used for generating predictions.
- **model_info.json**: A JSON file describing the architecture of the neural network model.
- **README.md**: This file, providing an overview and instructions for the project.

## Prerequisites
To run the project, you need the following dependencies:
- **Python 3.x**
- **Required Libraries**:
  ```bash
  pip install pandas numpy scikit-learn tensorflow category_encoders
  ```
- **Input Files**:
  - `petfinder_train.csv`: The training dataset.
  - `petfinder_test.csv`: The test dataset.
  Ensure these files are placed in a `data` directory relative to the notebook.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install the required Python packages:
   ```bash
   pip install pandas numpy scikit-learn tensorflow category_encoders
   ```
3. Ensure the `petfinder_train.csv` and `petfinder_test.csv` files are in the `data` directory.

## Usage
The project is executed within the `new_home.ipynb` Jupyter Notebook. Below is an overview of the workflow:

### 1. Data Loading
- The training and test datasets are loaded from `petfinder_train.csv` and `petfinder_test.csv` using **pandas**.
- The training dataset includes the `AdoptionSpeed` column, which is used to create the target variable.

### 2. Data Preprocessing
The preprocessing steps include:
- **Creating the Target Variable**:
  - A new `Target` column is created in the training dataset, where `True` indicates adoption (AdoptionSpeed < 4) and `False` indicates no adoption (AdoptionSpeed = 4).
- **Dropping Unnecessary Columns**:
  - The `AdoptionSpeed` and `Description` columns are dropped from the training dataset.
  - The `Description` column is dropped from the test dataset.
- **Encoding Categorical Features**:
  - Ordinal features (`MaturitySize`, `FurLength`, `Health`) are encoded using **LabelEncoder** from **scikit-learn**.
  - Nominal features (`Type`, `Gender`) are encoded using **LabelEncoder** (since they have only two categories).
  - Nominal features with multiple categories (`Breed1`, `Color1`, `Color2`, `Vaccinated`, `Sterilized`) are encoded using **BinaryEncoder** from **category_encoders** to avoid introducing artificial ordinality.
- **Normalization**:
  - Numerical features (`Age`, `Fee`, `PhotoAmt`) and encoded features are normalized to have a mean of 0 and a standard deviation of 1 using the mean and standard deviation from the training dataset.
- **Train-Validation Split**:
  - The training data is split into 90% training and 10% validation sets using **train_test_split** from **scikit-learn**.

### 3. Model Architecture
The neural network is built using **Keras** with the following architecture:
- **Input Layer**: Matches the number of features in the preprocessed dataset.
- **Hidden Layers**:
  - Dense layer with 5000 neurons and ReLU activation.
  - Dense layer with 1000 neurons and ReLU activation.
  - Dense layer with 500 neurons and ReLU activation.
- **Output Layer**: Single neuron with sigmoid activation for binary classification (True/False).
- **Total Parameters**: Approximately 5.6 million.

### 4. Model Training
- The model is compiled with:
  - Optimizer: **Adam**
  - Loss function: **BinaryCrossentropy**
  - Metric: **Accuracy**
- The model is trained for 10 epochs with a batch size of 128, using the validation set to monitor performance.

### 5. Prediction and Submission
- Predictions are generated for the test dataset using the trained model.
- Probabilities are converted to binary values (`True`/`False`) using a threshold of 0.5.
- The predictions are saved in a `submission.csv` file with a single `Target` column.

### Running the Notebook
To execute the project:
1. Open `new_home.ipynb` in Jupyter Notebook or JupyterLab.
2. Ensure the required libraries are installed and the input files are in the `data` directory.
3. Run all cells in the notebook to perform preprocessing, model training, and prediction.
4. The final output (`submission.csv`) will be generated, along with `test.csv` and `model_info.json`.

### Generating the Submission File
The notebook includes a cell to create a `result.zip` file containing:
- `submission.csv`: Predictions for the test dataset.
- `test.csv`: Preprocessed test dataset.
- `model_info.json`: Model architecture details.
- `new_home.ipynb`: The complete notebook.

Run the final cell to generate `result.zip`:
```python
import zipfile
import json
import os

# Save the notebook
if not os.path.exists(os.path.join(os.getcwd(), 'new_home.ipynb')):
    %notebook -e new_home.ipynb

# Save test dataset
test.to_csv("test.csv")

# Save model architecture
model_info = []
for layer in model.layers:
    if layer.__class__.__name__ == "Dense":
        model_info.append({
            "name": layer.__class__.__name__,
            "units": layer.units,
            "activation": layer.get_config()["activation"]
        })
    else:
        model_info.append({"name": layer.__class__.__name__})
with open("model_info.json", "w") as f:
    json.dump(model_info, f)

# Compress files
def compress(file_names):
    print("File Paths:")
    print(file_names)
    compression = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile("result.zip", mode="w") as zf:
        for file_name in file_names:
            zf.write('./' + file_name, file_name, compress_type=compression)

submission.to_csv('submission.csv', index=False)
file_names = ["test.csv", 'submission.csv', 'model_info.json', 'new_home.ipynb']
compress(file_names)
```

## Evaluation
The model's performance is evaluated using the **F1 score** with the `"weighted"` averaging method, as implemented in **scikit-learn**. The target is to achieve an F1 score of at least 70 to meet the project's requirements.

## Results
- The model achieves an accuracy of approximately 75–80% on the validation set after 10 epochs, with a validation loss around 0.5–0.55.
- The `submission.csv` file contains binary predictions (`True`/`False`) for the test dataset, indicating whether each pet is predicted to be adopted.

## Notes
- The `Description` column is excluded from the analysis as text processing is not covered in this project.
- Preprocessing steps (e.g., normalization, encoding) are applied consistently to both training and test datasets to ensure compatibility.
- The model architecture is relatively large (5.6M parameters), which may lead to overfitting on smaller datasets. Future improvements could include adding regularization (e.g., dropout) or tuning hyperparameters.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss improvements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.