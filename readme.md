# **Fall Detection with Custom Dataset Using Edge Impulse**

## **Overview**

This project develops a fall-detection system using **inertial data collected from two ESP32-S3 devices** worn simultaneously on the **foot** and **wrist**.
A fully custom dataset was recorded in controlled conditions, processed in Edge Impulse, and used to train a TinyML model based on **spectral features + dense neural networks**.
The goal is to evaluate whether real-time fall detection is feasible from these sensor locations and to compare performance between them.

# **1. Custom Dataset Creation**

## **1.1 Experimental Setup**

Two ESP32-S3 devices were equipped with:

* 3-axis accelerometer
* 3-axis gyroscope (discarded later due to saturation issues)
* Custom 3D-printed enclosure for impact protection
* BLE streaming system (device → Python receiver → CSV)

Sensors were placed at:

* **Foot** (smart-insole prototype location)
* **Wrist** (baseline comparison point widely used in wearables)

Data collection was performed on a **tatami floor** at the University of Zaragoza sports facility to ensure safety.

## **1.2 Participants**

* **13 volunteers**
* Physical characteristics recorded (age, height, weight, gender)
* Each participant performed a fixed protocol with identical timings

## **1.3 Recorded Activities**

The protocol contains **33 activities**, divided into:

### **ADLs (17 activities, 43 repetitions, ~10 minutes per participant)**

Examples:

* Slow/fast walking
* Stair climbing (slow/fast)
* Sitting and standing transitions
* Jumping, stumbling, picking up objects
* Agacharse, small obstacle avoidance steps

### **Falls (16 activities, 52 repetitions, ~8.5 minutes per participant)**

Including:

* Forward, backward, lateral falls
* Falls from walking, trotting, sitting, or standing
* Falls caused by simulated slips, trips, loss of balance
* Desmayo-type vertical collapses

## **1.4 Dataset Size**

Across all participants:

* **1,235 total repetitions**
* **approx. 481 minutes of data recorded**
* **2,470 CSV files** (foot + wrist)
* ADLs produce more windows than falls due to longer duration

All data was synchronized, validated via sequence counters, and exported in timestamped CSV format.

# **2. Edge Impulse Pipeline**

The same ML pipeline was applied independently to the wrist and foot datasets.

## **2.1 Windowing Strategy**

The system uses sliding windows designed to cover the full duration of a fall event:

* **Window Size:** large enough to contain the entire fall episode without fragmentation
* **Rationale:** falls are short (2–3s), but segmenting them manually is impractical without video annotations; therefore long automatic windows are acceptable and avoid mislabeling transitions
* **Moderate overlap** ensures no fall is split across boundaries

## **2.2 Feature Extraction**

A **Spectral Features** block is used:

* **FFT length:** 64
* Provides sufficient frequency resolution to characterize abrupt, high-energy impulses typical of falls
* Computationally light enough for embedded execution

Gyroscope data was **excluded**:

* Manufacturer default settings produced angular-rate saturation during impacts (flat-topped traces)
* These saturated signals degraded model learning
* Only accelerometer channels were retained to ensure data reliability

## **2.3 Model Architecture**

A **Dense Neural Network (DNN)** with 3 layers:

* **64 neurons** (high-resolution spectral representation)
* **32 neurons** (mid-resolution aggregation)
* **16 neurons** (general class-level interpretation)
* **Softmax output** (FALL / ADL)

Training:

* **50 epochs**
* **Class weighting enabled** (addresses dominance of ADL windows)
* **Feature normalization enabled** (reduces inter-subject scaling differences)
* **Quantization to int8** for deployment efficiency

## **2.4 Class Balancing**

Although falls and ADLs have similar *counts*, ADLs last longer and generate **far more windows**.
To avoid bias toward ADLs:

* **Class weighting** was activated
* Result: substantial reduction in fall false-negatives
* Trade-off: slightly more false positives acceptable in a safety-critical system

## **2.5 Validation Strategy**

A **leave-one-subject-out** validation approach was used:

* For each run, one participant was **completely excluded** from training
* That participant served as the **test subject**
* Prevents temporal leakage between training/testing
* Accurately simulates real deployment (new unseen users)

This approach significantly increases reliability compared to random window-shuffling.

# **3. Results**

## **3.1 Wrist Placement**

* **Accuracy:** ~98.6 %
* **Weighted Precision / Recall / F1:** ≈ 0.99
* **Recall for FALL:** ~100 %
* **Behavior:** No falls missed; a small fraction of ADLs misclassified as FALL
* Indicates high sensitivity and excellent generalization to unseen participants

## **3.2 Foot Placement**

* **Accuracy:** ~96.5 %
* **Weighted metrics:** 0.96–0.97
* **Recall for FALL:** ~94 %
* A small number of falls were misclassified as ADLs
* Still strong performance, confirming viability of a smart-insole system

## **3.3 Comparative Interpretation**

* Both locations achieve **high real-world feasibility**
* Wrist: slightly superior due to higher signal amplitude and variability
* Foot: slightly more confusion in transitions, but ideal for wearable integration

# **4. Discussion**

### **Key insights**

* Long windows are crucial for robust fall detection
* FFT=64 provides ideal balance of detail and computational cost
* Gyroscope removal improved overall consistency
* Normalization + class weighting = large improvement in fall sensitivity
* Subject-wise validation is mandatory to evaluate generalization

### **Safety considerations**

In fall detection, **recall for FALL** is the primary metric.
Both models maintain high recall, with the wrist achieving near-perfect performance.

# **5. Conclusion**

This project demonstrates that:

* A fully custom dataset collected with ESP32-S3 devices is **sufficiently rich** to train reliable fall detectors.
* Edge Impulse’s spectral + dense neural network pipeline can achieve **96–99 % accuracy**, depending on sensor location.
* Both wrist and foot placements are viable for real-world deployment.
* The final pipeline is efficient enough for **TinyML deployment on ESP32-S3**.
