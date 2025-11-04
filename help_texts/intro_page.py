welcome_help = """### Welcome to Mycoscope

**Mycoscope** is an interactive platform for exploring, segmenting, and classifying microscopy images — powered by AI and designed to streamline your analysis workflow.

With Mycoscope, you can:

- **Upload microscopy images** of cells, tissues, or other biological samples.  
- **Add or generate segmentation masks** using built-in AI tools such as **Cellpose** or **SAM2**.  
- **Classify cells** automatically with trained models (e.g., **DenseNet**) or manually by selecting cells in the image.  
- **Review and refine your results** interactively — correct masks, relabel cells, or adjust classifications in real time.  
- **Download your processed datasets** for external analysis, or  
- **Train new machine learning models** directly within Mycoscope to accelerate future analyses.

Whether you’re preparing data for quantitative cell analysis, building custom AI models, or exploring sample morphology, Mycoscope gives you a flexible, visual environment to make the process fast, transparent, and reproducible.
"""

upload_help = """### 1. Upload Your Data

Welcome to the **Upload page** — this is where your analysis begins.

Here you can upload the key files **Mycoscope** will use in later steps:

- **Images** *(required)* – the microscopy or sample images you want to analyze.  
- **Masks** *(optional)* – segmentation masks that outline cells or regions of interest.  
- **Cellpose model** *(optional)* – a trained model you can reuse for automatic cell segmentation.  
- **DenseNet model** *(optional)* – a classification model for identifying or labeling the segmented cells.

Once your files are uploaded, you’ll see a **summary table** showing:
- Which images have masks linked,  
- How many cells were detected in each image,  
- And any models you’ve provided for analysis or classification.

Use this page to confirm that all your data and models are correctly loaded before moving on to image analysis.
"""

segmentclassify_help = """### 2. Segmentation & Classification

This page is where you can **explore, edit, and analyze your images** using Mycoscope’s AI and interactive tools.

Here you can:

- **View your uploaded images** overlaid with their associated **cell masks**.  
- **Generate new masks** automatically using AI models such as **Cellpose** or **SAM2**.  
- **Manually edit or correct masks** with interactive tools — add, remove, or adjust individual cells as needed.  
- **Classify cells** using:  
  - an uploaded **DenseNet model** for automated classification, or  
  - **manual labeling** by clicking directly on cells in the image.  

Once your masks and classifications are ready, you can:

- **Download your dataset** for use in your own analysis pipelines, or  
- **Train new machine learning models** within Mycoscope to automate future segmentation and classification tasks.

Use this page to refine your data and build smarter, faster workflows for your cell analysis.
"""
