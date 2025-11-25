# **Fall Detection with Custom Dataset Using Edge Impulse**

## **Overview**

This project develops a fall-detection system using **inertial data collected from two ESP32-S3 devices** worn simultaneously on the **foot** and **wrist**.
A fully custom dataset was recorded in controlled conditions, processed in Edge Impulse, and used to train a TinyML model based on spectral features and dense neural networks.
The goal is to evaluate whether real-time fall detection is feasible from these sensor locations and to compare performance between them.

---

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

---

# **2. Edge Impulse Pipeline**

## **2.1 Windowing & Segmentation**

To automate processing of thousands of files, the system uses **large sliding windows** covering the entire duration of a fall event.

Ideally, each fall would be **manually trimmed to its exact duration** (2–3 seconds), but this would require synchronized video, frame-by-frame review and re-labelling.
Due to time and resource constraints—and because the model already behaved reliably—this manual segmentation was not feasible.
Instead, long windows (with moderate overlap) preserved entire fall episodes without splitting them, avoiding mislabeled transitions.

## **2.2 Feature Extraction**

A **Spectral Features** block was employed:

* **FFT length = 64**
* Balanced spectral resolution and embedded efficiency

### **Gyroscope exclusion**

During early sessions, gyroscope readings showed **saturation and flat-topped plateaus** due to the default angular-rate range and high impact velocities.
This corrupted impact shape and degraded learning, so **gyroscope data was excluded**, using only accelerometer channels for detection.

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

## **2.5 Edge Impulse Projects**

The full pipelines (dataset, impulse design, and trained models) are publicly available in Edge Impulse:

* **Wrist project:**
  [https://studio.edgeimpulse.com/studio/788227](https://studio.edgeimpulse.com/studio/788227)

* **Foot project:**
  [https://studio.edgeimpulse.com/studio/788229](https://studio.edgeimpulse.com/studio/788229)

---

# **3. On-Device Implementation**

Once trained, the Edge Impulse models were deployed as a **C++ library** on an ESP32-S3 with a **QMI8658 IMU**.
The firmware is the same for both locations (wrist/foot), changing only the embedded model.

### **3.1 Sampling and Inference**

* The accelerometer is sampled at **`EI_CLASSIFIER_FREQUENCY` Hz**, limited to **±4 g** and converted to **m/s²**.
* A sliding window of **497 samples × 3 axes** feeds the Edge Impulse classifier (`FallDetection_Wrist` / foot variant).
* Inference is **adaptive**:

  * In normal conditions, the model runs roughly every `WINDOW/8` samples.
  * When a potential event is armed, it switches to denser windows (`WINDOW/10`) for faster decisions.

### **3.2 Fall Probability and State Machine**

* Each inference returns a **fall probability** `p(FALL)`.

* A simple **exponential moving average** is maintained:

  `fall_ema = 0.25·p + 0.75·fall_ema`

* A compact state machine processes `p` and `fall_ema`:

  * **Idle** → arms an episode when `p ≥ 0.95` and `fall_ema ≥ 0.90`.
  * **AwaitHold** → in the next windows, requires consistent evidence (e.g. ≥2/3 windows with `fall_ema ≥ 0.70`) or cancels and returns to Idle.
  * **AwaitLow** → checks post-impact behavior:

    * Confirms a fall if several windows show **low probability** and **low acceleration variability** (impact + quietness).
    * If probability remains very high without quietness, it triggers a **“Need help?”** prompt instead.
  * **AwaitHelp** → handles the touchscreen prompt (YES/NO or auto-YES after a short timeout) and applies a **refractory period (~15 s)** to avoid repeated alarms.

### **3.3 Alerts and Telemetry**

* Confirmed events turn the screen **red** and optionally activate vibration.
* The device can **stream probabilities, EMA and state information via BLE notifications**, allowing external dashboards to monitor the detector in real time.

---

# **4. Results**

## **4.1 Wrist**

* **Accuracy:** ~98.6%
* **Weighted F1:** ~0.99
* **Recall (FALL):** ~100%
* No missed falls; minimal ADL confusion.

## **4.2 Foot**

* **Accuracy:** ~96.5%
* **Weighted F1:** 0.96–0.97
* **Recall (FALL):** ~94%
* Small number of falls misclassified as ADLs.

## **4.3 Interpretation**

* Both placements are **feasible** for real-time fall detection.
* The **wrist** shows slightly higher sensitivity and fewer missed falls.
* The **foot** remains attractive for smart-insoles, rehabilitation and daily comfort use-cases.
* Differences are not large enough to discard either location; choice can be driven by **ergonomics and clinical context**.

---

# **5. Discussion**

### **Main takeaways**

* Long windows prevent fragmentation of fall events.
* FFT = 64 is an appropriate trade-off for resolution and computational cost.
* Excluding gyroscope channels improved robustness due to saturation artefacts.
* Normalization and class weighting significantly boosted fall detection performance.
* Subject-wise validation is essential for realistic evaluation of generalization.
* The on-device state machine plus EMA converts raw model predictions into a **practical alert logic**.

### **Safety priority**

In fall detection, **recall for FALL** is the most important metric.
Both models excel, especially the wrist placement, and the firmware is biased towards **avoiding missed falls**, while keeping false alarms manageable through thresholds, post-impact checks and user confirmation.

---

# **6. Conclusion**

A custom fall dataset collected using ESP32-S3 devices is sufficient to train **high-performance TinyML models** in Edge Impulse.
Using spectral features and a 3-layer dense network, models achieved:

* **~99% accuracy** (wrist)
* **~96% accuracy** (foot)
* High recall for fall events in both locations

The combination of:

* a robust Edge Impulse pipeline,
* an adaptive inference schedule, and
* a lightweight on-device state machine with EMA and touchscreen prompts

shows that **real-time fall detection on low-power microcontrollers is feasible and practical** for wearable and smart insole applications.
