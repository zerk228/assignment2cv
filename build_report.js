const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ImageRun, AlignmentType, PageBreak, ShadingType, BorderStyle,
} = require("docx");

const { imageSize } = require("image-size");

function img(path, width) {
  const data = fs.readFileSync(path);
  const dims = imageSize(data);
  const h = Math.round(width * (dims.height / dims.width));
  return new ImageRun({ data, transformation: { width, height: h }, type: "png" });
}

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1 });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2 });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 120 } });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 80 } });
}
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, size: 18, color: "555555" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
  });
}
function centerImage(path, width) {
  return new Paragraph({ children: [img(path, width)], alignment: AlignmentType.CENTER, spacing: { after: 80 } });
}

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 2000, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "DCE6F1" } : undefined,
    children: [new Paragraph({ children: [new TextRun({ text, bold: !!opts.header })] })],
  });
}

const metricsRows = [
  ["Model", "Accuracy", "Precision", "Recall", "Macro F1", "Avg latency (ms)"],
  ["A1 Baseline (Logistic Regression)", "0.883", "0.884", "0.883", "0.883", "0.006"],
  ["A1 CNN", "0.990", "0.990", "0.990", "0.990", "0.528"],
  ["A2 ViT - no mixup", "0.919", "0.922", "0.919", "0.919", "0.208"],
  ["A2 ViT - with mixup (ablation)", "0.908", "0.915", "0.908", "0.908", "0.186"],
];

const metricsTable = new Table({
  width: { size: 9500, type: WidthType.DXA },
  columnWidths: [3000, 1300, 1300, 1300, 1300, 1300],
  rows: metricsRows.map((row, i) =>
    new TableRow({
      children: row.map((text, j) => cell(text, { header: i === 0, width: [3000, 1300, 1300, 1300, 1300, 1300][j] })),
    })
  ),
});

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Calibri", size: 22 } },
      heading1: { run: { font: "Calibri", size: 30, bold: true, color: "1F4E79" }, paragraph: { spacing: { before: 200, after: 100 }, keepNext: true } },
      heading2: { run: { font: "Calibri", size: 24, bold: true, color: "2E5496" }, paragraph: { spacing: { before: 150, after: 80 }, keepNext: true } },
      title: { run: { font: "Calibri", size: 44, bold: true }, paragraph: { spacing: { before: 0, after: 200 }, keepNext: true } },
    },
  },
  sections: [
    {
      properties: { page: { size: { width: 12240, height: 15840 } } },
      children: [
        new Paragraph({ text: "Assignment 2 Report", heading: HeadingLevel.TITLE, alignment: AlignmentType.CENTER }),
        new Paragraph({
          children: [new TextRun({ text: "Level-Up Your A1 Topic with SOTA CV + Demo", bold: true, size: 28 })],
          alignment: AlignmentType.CENTER, spacing: { after: 100 },
        }),
        new Paragraph({
          children: [new TextRun({ text: "Topic: Handwritten Digits and Symbols (0-9, plus, minus)", size: 24 })],
          alignment: AlignmentType.CENTER, spacing: { after: 400 },
        }),
        new Paragraph({ children: [new TextRun({ text: "Astana IT University - Computer Vision", size: 22, color: "555555" })], alignment: AlignmentType.CENTER }),
        new Paragraph({ children: [new TextRun({ text: "Instructor: Nurym Kaiyrkhan", size: 22, color: "555555" })], alignment: AlignmentType.CENTER }),
        new Paragraph({ children: [new PageBreak()] }),

        h1("1. Purpose and Overview"),
        p("Assignment 1 built a baseline recognizer for hand-drawn digits (0-9) and two math symbols (plus, minus) using a logistic regression baseline and a small convolutional neural network (CNN). Assignment 2 keeps the same topic and dataset and upgrades the pipeline with a state-of-the-art (SOTA) computer-vision component: a Vision Transformer (ViT) classifier with attention-based explainability, delivered through an interactive web demo."),
        p("The goals of this continuation are to: (1) add a SOTA method and compare it against the A1 baseline, (2) ship a small interactive app with a visual explanation of each prediction, (3) report metrics, training curves and an ablation study, and (4) reflect on ethics and reproducibility."),

        h1("2. Dataset Update"),
        bullet("12 balanced classes: digits 0-9, 'plus', 'minus'."),
        bullet("9,600 total 64x64 grayscale images - 640 train / 160 test images per class, stratified split (80/20)."),
        bullet("Augmentation: for every original hand-drawn image, 3 synthetic variants are generated (random rotation +/-15 deg, brightness jitter 0.8-1.2x, Gaussian pixel noise sigma=0.02). This means 75% of the training images are synthetic augmentations and 25% are original drawings."),
        bullet("No new raw images were collected for Assignment 2; the same A1 dataset and preprocessing (prepare_dataset.py) were reused unchanged, as allowed by the assignment brief."),

        h1("3. SOTA Method: Vision Transformer (ViT)"),
        p("For the Beginner track (handwritten digits/symbols), we selected a modern, practical SOTA architecture for image classification - the Vision Transformer (Dosovitskiy et al., \"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale\", ICLR 2021) - instead of the heavier generative pipelines (TA-TiTok/MaskGen/ArtAug) suggested for data synthesis, since those target large-scale photographic datasets and would not meaningfully improve a small, clean, already-balanced digit dataset. ViT is a widely recognized SOTA family for image recognition and, unlike the A1 CNN, it produces a native attention map that can be used directly for explainability (attention rollout), matching the assignment's visual-explanation requirement."),
        h2("3.1 Architecture"),
        p("Each 64x64 grayscale image is split into 8x8 patches (64 patches total), linearly embedded to a 64-dimensional vector, prepended with a learnable [CLS] token, and combined with learnable positional embeddings. Four transformer encoder blocks (multi-head self-attention + MLP, 4 heads) process the sequence; the final [CLS] token representation is passed to a linear classification head."),
        centerImage("reports/vit_diagram.png", 560),
        caption("Figure 1. ViT architecture used for Assignment 2 (patch embedding -> CLS+positional embedding -> 4x transformer blocks -> linear head)."),
        h2("3.2 Why it helps this topic"),
        p("Digits and symbols are defined by stroke shape and relative position of strokes (e.g. the crossbar of a plus sign). Self-attention lets every patch directly attend to every other patch from the first layer, which is well suited to capturing these long-range stroke relationships, and it naturally yields a per-image attention map that shows exactly which regions drove the decision - something the A1 CNN cannot provide without extra instrumentation (Grad-CAM)."),

        h1("4. Experiments"),
        h2("4.1 Metrics: A1 baseline vs A2 SOTA"),
        metricsTable,
        new Paragraph({ text: "", spacing: { after: 150 } }),
        p("The A1 CNN remains the strongest model on this dataset (99.0% accuracy) because convolutional inductive bias (locality + weight sharing) is very effective on a small, clean, centered digit dataset with limited data. The ViT reaches 91.9% without mixup - substantially better than the linear baseline (88.3%) - demonstrating that the SOTA attention-based architecture is competitive even without the convolutional prior, while adding transparent attention-based explainability that the CNN baseline lacks."),
        centerImage("reports/metrics_comparison_a2.png", 440),
        caption("Figure 2. Accuracy and macro-F1 across all four models."),

        h2("4.2 Training curves"),
        centerImage("reports/training_curves_combined.png", 480),
        caption("Figure 3. ViT training loss per epoch, without mixup (left) and with mixup (right)."),

        h2("4.3 Ablation: Mixup augmentation"),
        p("We ran an ablation comparing the ViT trained with and without mixup augmentation (convex blending of image pairs and their one-hot labels during training, alpha=0.4). Mixup is normally used to reduce overfitting and improve robustness in low-data regimes."),
        bullet("Without mixup: accuracy 0.919, macro-F1 0.919."),
        bullet("With mixup: accuracy 0.908, macro-F1 0.908."),
        p("On this dataset, mixup slightly reduced accuracy. The dataset is small, already balanced and already augmented (rotation/brightness/noise), and digit/symbol classes have very distinct shapes with little ambiguity - mixing pairs of different classes mainly adds label noise here rather than useful regularization. This is a useful negative result: mixup is not universally beneficial and should be validated per dataset rather than applied by default."),

        h1("5. Explainability"),
        p("The deployed app visualizes attention rollout from the ViT: attention weights from all 4 transformer blocks are recursively combined (with a residual identity term) to estimate how much each image patch influences the [CLS] token used for classification, then upsampled to the image resolution and overlaid as a heatmap."),
        centerImage("reports/app_screenshot.png", 420),
        caption("Figure 5. Live app screenshot - an uploaded '1' is correctly classified (83.3% confidence, 12 ms latency) with the attention heatmap concentrated on the digit stroke."),
        p("For comparison, Grad-CAM was also implemented for the A1 CNN baseline (explain.py), and a nearest-neighbor retrieval gallery was built from ViT [CLS] embeddings to show visually similar training examples for a given query image."),
        centerImage("reports/nearest_neighbors_gallery.png", 560),
        caption("Figure 6. Nearest-neighbor gallery: a query image (left) and its 5 closest matches in ViT embedding space, with their L2 distances."),

        h1("6. App (Web Demo)"),
        p("A small Flask web app (app.py) implements the required flow: upload an image -> ViT prediction -> attention-rollout heatmap overlay. Each result also reports average inference latency and the runtime device/OS, as required by the assignment."),
        bullet("Core flow: upload -> prediction -> visual explanation (attention heatmap)."),
        bullet("Latency and device/environment shown for every prediction."),
        bullet("No heavy external generators were used, so no offline-generation step or synthetic-media labeling was required for the app itself; the dataset-level synthetic augmentation share is disclosed in Section 2."),

        h1("7. Ethics and Limitations"),
        bullet("Dataset: all images are hand-drawn digits/symbols created by the project author for this course; no third-party personal data, faces, or biometric information is involved."),
        bullet("Bias: the dataset was collected by a single author, so handwriting style diversity is limited; performance on different handwriting styles, left-handed writers, or other scripts is not guaranteed and was not tested."),
        bullet("Misuse risk: this classifier is narrow-purpose (digits/symbols) and has no realistic misuse potential; the app includes an on-screen disclaimer that predictions come from a model trained on student-collected data for educational purposes."),
        bullet("Known limitation: the ViT underperforms the A1 CNN on this specific small dataset (Section 4.1); it should not be presented as a strict upgrade in raw accuracy, only as a SOTA architecture that adds native attention-based explainability."),

        h1("8. Reproducibility"),
        p("Environment: Python 3.9, PyTorch 2.8 (CPU), scikit-learn, Flask, matplotlib, Pillow (see requirements.txt)."),
        bullet("1. pip install -r requirements.txt"),
        bullet("2. python eval_baseline.py  (evaluates A1 logistic-regression + CNN weights, already included under models/)"),
        bullet("3. python train_sota.py  (trains the ViT with and without mixup, saves weights to models/vit_no_mixup.pt and models/vit_mixup.pt and metrics to reports/vit_metrics.json)"),
        bullet("4. python nearest_neighbors_demo.py and python plot_comparison.py  (regenerate the figures used in this report)"),
        bullet("5. python app.py  (starts the demo at http://localhost:5000)"),
        p("All trained weights, dataset tensors, and generated figures are included in the submitted repository under models/, data/, and reports/."),

        h1("9. Conclusion"),
        p("Assignment 2 successfully extends the A1 handwritten digits/symbols project with a Vision Transformer SOTA component, native attention-based explainability, a working interactive web demo, and an ablation study on mixup augmentation. While the A1 CNN remains the most accurate model on this small dataset, the ViT provides a transparent, attention-driven alternative and the overall pipeline now reports accuracy, latency, and explainability for every prediction, directly addressing the assignment's requirements."),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync("report.docx", buffer);
  console.log("report.docx written");
});
