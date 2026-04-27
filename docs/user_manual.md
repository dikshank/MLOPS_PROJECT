# User Manual — MelanoCheck
## Early Melanoma Detection Tool

---

## What is MelanoCheck?

MelanoCheck is a simple tool that looks at a photo of a skin lesion and
tells you whether it might be melanoma (a type of skin cancer) or benign
(not cancer). It is designed to work even with low-quality photos taken
on basic mobile phones.

**Important:** This tool is for screening only. It does not replace a
doctor's diagnosis. Always consult a qualified medical professional if
you have any concerns about a skin lesion.

---

## Getting Started

Open your web browser (Chrome, Firefox, or Safari) and go to:
```
http://localhost:80
```

You will see the MelanoCheck home screen with two tabs at the top:
- **Screening** — upload images and get predictions
- **Pipeline Dashboard** — view live system metrics and pipeline status

---

## Tab 1: Screening

### Step 1 — Upload your image

You can upload an image in two ways:

**Option A — Click "Choose Image"**
1. Click the "Choose Image" button
2. A file browser will open
3. Select a photo of the skin lesion (JPEG or PNG format)
4. The photo will appear on the screen

**Option B — Drag and Drop**
1. Find the photo on your computer
2. Drag it onto the upload area on the screen
3. Drop it — the photo will appear

**Tips for a good photo:**
- Make sure the lesion is clearly visible
- Good lighting helps, but low-quality photos also work
- Avoid blurry images where possible

---

### Step 2 — Click "Analyse Image"

Once your photo is uploaded, click the **"Analyse Image"** button.

The tool will analyse the photo. This usually takes 1-2 seconds.

---

### Step 3 — Read your result

The result will appear below the upload area.

**If the result shows:**

**Malignant Detected (amber/orange background)**
> The tool has found signs that may indicate melanoma.
> Please consult a dermatologist or medical professional as soon as possible.
> This does not mean you definitely have cancer — a doctor must confirm.

**Benign — Low Risk (blue background)**
> The tool has not found strong signs of melanoma.
> However, if you have any concerns, please still consult a doctor.
> Regular skin checks are always recommended.

You will also see:
- **Confidence** — how certain the tool is (higher is more certain)
- **Malignant Probability** — raw probability score from the model
- **Threshold Used** — the decision boundary applied
- **Recommendation** — personalised advice on what to do next

---

### Step 4 — Provide feedback (medical professionals only)

If you are a medical professional and know the actual diagnosis:
- Click **"Confirm: Malignant"** — if the lesion is actually malignant
- Click **"Confirm: Benign"** — if the lesion is actually benign

This feedback is saved securely and used to improve the model over time.
Note: the image is saved on the local server for retraining purposes.

---

### Step 5 — Analyse another image

Click **"Analyse Another Image"** to reset and upload a new photo.

---

### System Status Panel

At the bottom of the Screening tab you will see:
- **API Server** — blue dot means the system is running, orange means offline
- **Model** — blue dot means the AI model is ready
- Links to Airflow, MLflow, and Grafana dashboards

---

## Tab 2: Pipeline Dashboard

Click **"Pipeline Dashboard"** at the top to see the system's internal state.

### Live Pipeline Metrics

Shows real-time numbers updated every 15 seconds:
- **Total Requests** — how many API calls have been made
- **Predictions** — how many images have been analysed
- **Feedback Received** — how many feedback submissions
- **Drift Score** — how different incoming images are from training data
- **Real-World Recall** — percentage of actual melanomas correctly detected
- **Misclassification Rate** — percentage of wrong predictions from feedback

The **Drift Score bar** shows current drift with a threshold line at 20%.
If drift exceeds 20%, the system automatically triggers retraining.

### ML Pipeline Stages

Shows the complete pipeline from raw data to serving:
```
Raw Data → Airflow DAG → Training → Registry → Serving
```

### Pipeline Management Tools

Click any tool card to open it in a new tab:
- **Airflow** — view DAG runs, task logs, scheduling
- **MLflow** — view experiments, model registry, training artifacts
- **Grafana** — view NRT monitoring dashboards
- **Prometheus** — query raw metrics

### Current Production Model

Shows which model version is currently serving predictions:
- Model name (e.g. simple_cnn)
- Registry version number
- Classification threshold
- Status (ready/not_ready)

---

## Frequently Asked Questions

**Q: What image formats are supported?**
A: JPEG (.jpg, .jpeg) and PNG (.png) files are supported.

**Q: My photo is blurry or low quality. Will it still work?**
A: Yes. MelanoCheck is specifically designed to work with low-quality
images from basic mobile cameras (32x32 pixel resolution and above).

**Q: How accurate is the tool?**
A: The tool is optimised to catch as many potential melanoma cases as
possible (high recall). This means it may sometimes flag benign lesions
as potentially malignant. This is intentional — it is better to consult
a doctor unnecessarily than to miss a real cancer.

**Q: Is my photo stored?**
A: Yes. Photos are saved temporarily on the local server in a secure
folder for potential model retraining purposes. They are not sent to
any external server or cloud service.

**Q: What should I do if the tool says "Malignant"?**
A: Do not panic. The tool is a screening aid, not a diagnosis.
Contact a dermatologist or skin cancer clinic for a professional evaluation.

**Q: The page shows "Model not loaded". What does that mean?**
A: The AI model is still loading. Please wait 30 seconds and refresh
the page. If the problem persists, contact your system administrator.

**Q: What is the Drift Score?**
A: The drift score measures how different incoming images are from the
images used to train the model. A high drift score (above 20%) means
the model may need retraining on newer data. The system handles this
automatically.

**Q: What happens when the Misclassification Rate is too high?**
A: When more than 10% of predictions are wrong (based on doctor feedback),
the system automatically schedules a retraining run. The model is updated
and the best new model is promoted to production automatically.

---

## Colour Accessibility Note

MelanoCheck uses colours that are distinguishable by people with colour
blindness (deuteranopia/protanopia):
- Malignant results: amber/orange background with triangle symbol (▲)
- Benign results: blue background with circle symbol (●)
- Online status: blue dot
- Offline status: orange dot

---

## Contact and Support

If you experience any issues, please contact your system administrator.

---

*MelanoCheck is an AI-assisted screening tool built as part of an MLOps
research project. It is not a certified medical device.*
