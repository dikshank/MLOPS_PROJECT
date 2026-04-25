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

## How to Use MelanoCheck

### Step 1 — Open the application

Open your web browser (Chrome, Firefox, or Safari) and go to:
```
http://localhost:80
```

You will see the MelanoCheck home screen.

---

### Step 2 — Upload your image

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

### Step 3 — Click "Analyse Image"

Once your photo is uploaded, click the green **"Analyse Image"** button.

The tool will analyse the photo. This usually takes a few seconds.

---

### Step 4 — Read your result

The result will appear below the upload area.

**If the result shows:**

🔴 **Malignant Detected**
> The tool has found signs that may indicate melanoma.
> Please consult a dermatologist or medical professional as soon as possible.
> This does not mean you definitely have cancer — a doctor must confirm.

🟢 **Benign — Low Risk**
> The tool has not found strong signs of melanoma.
> However, if you have any concerns, please still consult a doctor.
> Regular skin checks are always recommended.

You will also see:
- **Confidence** — how certain the tool is about its result (higher is more certain)
- **Recommendation** — advice on what to do next

---

### Step 5 — Provide feedback (optional, for medical professionals only)

If you are a medical professional and you know the actual diagnosis,
you can click either:
- **"Confirm: Malignant"** — if the lesion is actually malignant
- **"Confirm: Benign"** — if the lesion is actually benign

This feedback helps improve the tool over time.

---

### Step 6 — Analyse another image

Click **"Analyse Another Image"** to reset and upload a new photo.

---

## Frequently Asked Questions

**Q: What image formats are supported?**
A: JPEG (.jpg, .jpeg) and PNG (.png) files are supported.

**Q: My photo is blurry or low quality. Will it still work?**
A: Yes. MelanoCheck is specifically designed to work with low-quality
images from basic mobile cameras.

**Q: How accurate is the tool?**
A: The tool is optimised to catch as many potential melanoma cases as
possible (high recall). This means it may sometimes flag benign lesions
as potentially malignant. This is intentional — it is better to consult
a doctor unnecessarily than to miss a real cancer.

**Q: Is my photo stored anywhere?**
A: No. Photos are processed immediately and are not stored on any server.

**Q: What should I do if the tool says "Malignant"?**
A: Do not panic. The tool is a screening aid, not a diagnosis.
Contact a dermatologist or skin cancer clinic for a professional evaluation.

**Q: The page shows "Model not loaded". What does that mean?**
A: The AI model is still loading. Please wait 30 seconds and refresh the page.
If the problem persists, contact your system administrator.

---

## System Status

At the bottom of the page you will see a system status panel showing:
- **API Server** — green dot means the system is running
- **Model** — green dot means the AI model is ready
- Links to the technical dashboards (for system administrators)

---

## Contact and Support

If you experience any issues, please contact your system administrator.

---

*MelanoCheck is an AI-assisted screening tool built as part of an MLOps
research project. It is not a certified medical device.*
