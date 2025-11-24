workflow_help = """### Mycoscope Workflow Overview"""

welcome_help = """### Welcome to Mycoscope

**Mycoscope** is an interactive platform for exploring, segmenting, and classifying microscopy images — powered by AI and designed to streamline your analysis workflow.

With Mycoscope, you can:

- **Upload microscopy images** of cells or tissues.  
- **Add or generate segmentation masks** using built-in AI tools such as **Cellpose** or **SAM2**.  
- **Classify cells** automatically with trained **DenseNet** models or manually by selecting cells in the image.  
- **Review and refine your results** interactively — correct masks, relabel cells, or adjust classifications in real time.  
- **Download your processed datasets** for external analysis, or  
- **Train new machine learning models** directly within Mycoscope to accelerate future analyses.

Whether you’re preparing data for quantitative cell analysis, building custom AI models, or exploring sample morphology, Mycoscope gives you a flexible, visual environment to make the process fast, transparent, and reproducible.
"""

upload_help = """### 1. Upload Your data

Welcome to the **Upload page** — this is where your analysis begins.

Here you can upload the key files **Mycoscope** will use in later steps:

- **Images** *(required)* – the microscopy or sample images you want to analyze.  
- **Masks** *(optional)* – segmentation masks that outline cells or regions of interest.  
- **Cellpose model** *(optional)* – a trained model you can reuse for automatic cell segmentation.  
- **DenseNet model** *(optional)* – a classification model for identifying or labeling the segmented cells.

Once your files are uploaded, you’ll see a **summary table** showing:
- Which images have masks linked,  
- How many cells are highlighted in each image,  
- And any models you’ve provided for analysis or classification.

Use this page to confirm that all your data and models are correctly loaded before moving on to image analysis.
"""

segmentclassify_help = """### 2. Segment and classify your cells

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

train_help = """### 3. Train your own analysis models

This page allows you to train new machine learning models using the datasets you’ve uploaded or created on the **Segmentation & Classification** page.

You can choose to train either:

- **A Cellpose segmentation model** – to improve or customize how cells are automatically detected and outlined in your images.  
- **A DenseNet classification model** – to fine-tune how cells are categorized and labeled based on their features.

By default, Mycoscope provides sensible training parameters suitable for most datasets.  
However, you can adjust these parameters and run **hyperparameter optimization** to explore how different settings affect model performance.

After training completes, the app displays a set of **performance plots** that show:

- How the training progressed over time  
- How well the model fits and generalizes to your data  
- Metrics such as accuracy, loss, and validation performance

Once trained, you can:

- **Use the new model immediately** on the **Segmentation & Classification** page to analyze images automatically  
- **Download the trained model**, along with the training data and parameters, for reuse in publications or for future analyses

This page gives you a simple and reproducible way to build, evaluate, and deploy custom AI models tailored to your specific image data.
"""

analyze_help = """### 4. Get to know your data

The **Cell Analysis** page allows you to explore and summarize the quantitative results of your analyses.

You can create and download plots of **cell population statistics**—such as cell area, perimeter, and other morphological or intensity-based features—grouped by **cell class**.

Select which **classes** and **characteristics** to include, and Mycoscope will generate clear, publication-quality plots that help you:

- Compare how cell size, shape, or intensity differ between classes or conditions  
- Identify trends in cell populations across multiple images or experiments  
- Quantify variability and relationships among measured features  

In addition to plots, the downloadable results include:

- Tables summarizing the **number of cells per class** in each image  
- **Descriptive statistics** (mean, median, standard deviation, etc.) for all selected features  
- The generated plots in a ready-to-use format for reports or publications  

Use this page to transform your image-level analyses into meaningful biological insights about cell populations, morphology, and class-specific behaviors.
"""
