# SeizyML Pipeline Explained

SeizyML is designed for accurate detection of seizure events from EEG data using a combination of machine learning and manual verification. The pipeline consists of several key stages that process raw data, train the model, and refine predictions for reliable results.

<p align="center">
<center><img src="model_pipeline.png" width="600"></center>
</p>

---

#### 1. **Data Preparation**  
The process begins with raw LFP/EEG data. To make the data manageable for analysis:
- **Downsampling and Segmentation:** The data is divided into 5-second non-overlapping windows and downsampled to 100 Hz with an antialiasing filter, which helps maintain signal quality while reducing data size.
- **Format Conversion:** Data can be converted to the HDF5 format using **[SeizyConvert](https://github.com/neurosimata/seizy_convert)**, which currently supports conversion from the EDF format.

---

#### 2. **Data Preprocessing** (for both Training and Prediction)  
Preprocessing ensures the data is clean and suitable for analysis. This step is critical for both the model training and prediction phases:
- **High-Pass Filtering:** A high-pass filter at 2 Hz is applied to remove baseline drift and low-frequency noise.
- **Artifact Removal:** Extreme outliers (values >25 standard deviations) are identified and replaced with the median of the corresponding 5-second window to prevent distortion in the data.
---

#### 3. **Feature Extraction**  
SeizyML extracts 17 key features from each EEG channel to capture both statistical and spectral properties of the signal:
- **Time-Domain Features:** 
  - Line length, kurtosis, skewness, root mean square, median absolute deviation, variance, energy
- **Hjorth Parameters:** 
  - Hjorth mobility, Hjorth complexity
- **Frequency-Domain Features:** 
  - Delta power (2–4 Hz), theta power (4.2–8 Hz), alpha power (8.2–12 Hz), beta power (12.2–30 Hz), gamma power (30.2–50 Hz)
- **Other Features:** 
  - Weighted frequency, spectral entropy, envelope amplitude

Feature implementations are available in the [features.py](../seizyml/helper/features.py) script of the SeizyML repository.

---

#### 4. **Feature Selection**  
To enhance model performance and reduce redundancy:
- **Correlation Filtering:** Features with a Pearson correlation coefficient (r > 0.9) are removed to eliminate highly correlated variables.
- **ANOVA-Based Selection:** The top 5, 10, and 15 features are selected based on ANOVA comparisons with the target variable.
- **Diversity Enhancement:** Additional feature sets include the 5 least correlated (LC) features alongside the top features to increase informational diversity:
  - Top 5, Top 10, Top 15
  - Top 5 + LC (10 features), Top 10 + LC (15 features), Top 15 + LC (20 features)
- The feature set achieving the best **F1 score** during model evaluation is selected for final use.

---

#### 5. **Model Training**  
SeizyML uses a **Gaussian Naive Bayes (GNB)** model for seizure detection:
- **Cross-Validation:** Training is performed using **Stratified K-Fold Cross-Validation (K=5)** to ensure an even distribution of seizure and non-seizure events across training and validation sets.
- **Hyperparameter Tuning:** 
  - A grid search is used to optimize hyperparameters with **balanced accuracy** as the key performance metric.
  - For GNB, the `var_smoothing` parameter is tuned. These settings can be found in the [model_settings.py](../seizyml/train/model_settings.py) file.

#### 6. **Automated Seizure Detection (Prediction Phase)**  
After training, the model is applied to new EEG recordings:
- **Preprocessing:** The same steps from the training phase are repeated, including filtering, artifact removal, and segmentation to maintain consistency.
- **Feature Extraction:** Identical features are extracted to ensure compatibility with the model trained on the original dataset.
- **Prediction:** The trained **Gaussian Naive Bayes (GNB)** model generates predictions, identifying potential seizure events based on the extracted features.

---

#### 7. **Manual Verification (Using GUI)**  
SeizyML includes a graphical user interface (GUI) that allows users to review and refine the model’s predictions:
- **Event Review:** Users can manually inspect the detected seizure events and choose to accept or reject each prediction.
- **Boundary Refinement:** The start and end times of detected seizures can be adjusted to improve the accuracy of event marking, ensuring precise seizure annotation.

---

#### 8. **Data Analysis**  
SeizyML facilitates comprehensive data analysis after seizure events have been verified:
- **Seizure Metrics:** Extract key properties such as:
  - **Average Seizure Duration:** The mean length of seizure events across recordings.
  - **Seizure Frequency:** The number of seizures occurring within a defined timeframe.
- **Statistical Analysis:** Generate summary plots, conduct statistical tests, and compare seizure patterns across different datasets for in-depth analysis.

---

#### 9. **Feature Contribution Insights**  
To better understand the model’s decision-making process:
- **Feature Separation Score:** Visualizations are generated to display how different features contributed to the model’s ability to distinguish between seizure and non-seizure events, highlighting the most influential features in the prediction process.

**[<< Back to Main Page](/README.md)**
