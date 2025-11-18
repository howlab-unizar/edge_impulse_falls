# **Fall Detection with Custom Dataset Using Edge Impulse**

## **Overview**

This project develops a fall-detection system using **inertial data collected from two ESP32-S3 devices** worn simultaneously on the **foot** and **wrist**.
A fully custom dataset was recorded in controlled conditions, processed in Edge Impulse, and used to train a TinyML model based on spectral features and dense neural networks.
The goal is to evaluate whether real-time fall detection is feasible from these sensor locations and to compare performance between them.

# **1. Custom Dataset Creation**

## **1.1 Experimental Setup**

Two ESP32-S3 devices were equipped with:

* 3-axis accelerometer
* 3-axis gyroscope (later discarded due to saturation issues)
* Custom 3D-printed enclosure for impact protection
* BLE streaming system (device → Python receiver → CSV)

Sensors were worn on:

* **Foot** (smart-insole prototype location)
* **Wrist** (baseline comparison used in commercial wearables)

Experiments were conducted on a **tatami floor** at the University of Zaragoza to ensure safe fall simulation.

## **1.2 Participants**

* **13 volunteers**
* Different physical characteristics (age, weight, height, gender)

Each participant followed the exact same structured protocol.

## **1.3 Recorded Activities**

The protocol includes **33 activities**:

### **ADLs (17 activities, ~10 minutes per participant)**

Walking, stair climbing, sitting/standing transitions, jogging, jumping, stumbling, picking up objects…

### **Falls (16 activities, ~8.5 minutes per participant)**

Forward, backward, and lateral falls, falls while walking or sitting, simulated slips, trips, fainting-like vertical collapses, etc.

## **1.4 Dataset Size**

Across all sessions:

* **1,235 repetitions**
* **≈481 minutes recorded**
* **2,470 CSV files**
* ADLs dominate in number of windows due to their longer duration

All data was validated using sequence counters, timestamps, and connection monitoring.

# **2. Edge Impulse Pipeline**

## **2.1 Windowing & Segmentation**

To automate processing of thousands of files, the system uses **large sliding windows** covering the entire duration of a fall event.

### **Note on ideal segmentation**

Ideally, each fall would be **manually trimmed to its exact duration** (typically 2–3 seconds).
However, this requires:

* synchronized video recordings,
* manual review frame by frame,
* and per-window re-labeling.

Because of the time and resource constraints—and because the model already behaved reliably with automatic windows—this manual segmentation was **not feasible**.
Instead, long windows (with moderate overlap) preserved entire fall episodes without splitting them, avoiding mislabeled transitions.

## **2.2 Feature Extraction**

A **Spectral Features** block was employed:

* **FFT length = 64**
* Balanced spectral resolution and embedded efficiency

### **Gyroscope exclusion**

During early sessions, gyroscope readings showed **saturation and flat-topped plateaus** caused by:

* The default Waveshare configuration for angular-rate range
* High angular velocities during impact not fitting the configured range

This saturation corrupted the temporal shape of impacts, degrading model learning.

**Therefore, gyroscope data was excluded** and only accelerometer channels were used for detection.

## **2.3 Model Architecture**

Dense neural network:

* 64 → 32 → 16 neurons
* 50 epochs
* Normalization enabled
* Class weighting enabled to counter ADL dominance
* Post-training **int8 quantization** for ESP32-S3 deployment

## **2.4 Validation Strategy**

A **leave-one-subject-out** approach:

* One participant excluded entirely from training
* Used only as test subject
* Ensures generalization to unseen users
* Prevents window overlap leakage

# **3. Results**

## **3.1 Wrist**

* **Accuracy:** ~98.6%
* **Weighted F1:** ~0.99
* **Recall (FALL):** ~100%
* No missed falls; minimal ADL confusion

## **3.2 Foot**

* **Accuracy:** ~96.5%
* **Weighted F1:** 0.96–0.97
* **Recall (FALL):** ~94%
* Small number of falls misclassified as ADLs

## **3.3 Interpretation**

* Both placements feasible
* Wrist shows slightly higher sensitivity
* Foot remains attractive for smart-insoles and daily comfort
* Differences are not large enough to discard either location

# **4. Discussion**

### **Main takeaways**

* Long windows prevent fragmentation of fall events
* FFT=64 is an appropriate trade-off for resolution and cost
* Gyroscope exclusion improved robustness
* Normalization and class weighting significantly boosted fall detection
* Subject-wise validation is essential for realistic evaluation

### **Safety priority**

In fall detection, **recall for FALL** is the most important metric.
Both models excel, especially the wrist placement.

# **5. Conclusion**

A custom fall dataset collected using ESP32-S3 devices is sufficient to train **high-performance TinyML models** in Edge Impulse.
Using spectral features and a 3-layer dense network, models achieved:

* **~99% accuracy** (wrist)
* **~96% accuracy** (foot)
* High recall for fall events in both locations

Both configurations are viable for real-world deployment, with the choice driven by ergonomics and product design requirements.
