# Image Digitizer: Blueprint & Development Roadmap

## Part 1: Image to Embroidery Conversion Workflow (Hatch-Inspired Blueprint)

### 1. Artwork Preparation and Optimization
- **Optimal Sizing:** Crop the image to remove excess background. Size the image for its intended hoop purpose (optimal ~150mm x 150mm, limit ~210mm).
- **Image Quality:** Use high-resolution, clear photos or clean vector graphics with strong contrast.
- **Color Reduction:** Pre-process the image to reduce color depth (1–15 colors). Fewer colors mean cleaner paths and fewer thread changes on the machine.

### 2. Method A: Auto-Digitize Embroidery (For Clipart and Vectors)
- **Import Artwork:** Load prepared bitmap or vector image.
- **Color Classification:** Classify color regions as fills or details (outlines), or omit background colors.
- **Sort Stitch Sequence:** Order color layers (typically fills stitched first, outlines/details stitched last).
- **Assign Threads:** Match colors against thread charts (e.g., Madeira, Robison-Anton).
- **Process the Image:** Convert shapes into tatami, satin, and run stitches.

### 3. Method B: Color PhotoStitch (For Photographs)
- **Set Thread Count:** Limit thread count between 7 to 10 colors (max 15).
- **Resolution Settings:** Low, Medium, or High density variegated rows.
- **Match and Assign:** Map quantized color blocks to physical threads.

### 4. Method C: Click-to-Fill and Hybrid Workflows
- **Click-to-Fill:** Automatically fills a closed shape with a Tatami stitch while preserving interior holes.
- **Click-to-Turning Fill:** Converts narrow column shapes into satin stitches that dynamically follow the contour.
- **Click-to-Centerline:** Extracts the medial axis / skeleton of thin lines and converts it to a single running stitch.

### 5. Stitch Player Verification and Manual Refinements
- **TrueView & Stitch Player:** Simulate step-by-step stitch-out on screen to inspect jumps, density, and pathing.
- **Reshape and Edit:** Modify stitch types, density, stitch angles, or sewing sequence.

### 6. Exporting to the Embroidery Machine
1. **Save Source:** Native object-based design format retaining shapes, stitch types, colors, and properties.
2. **Export for Machine:** Convert object data into physical stitch coordinates (0.1mm machine units) for target formats (`.DST`, `.PES`, `.JEF`).

---

## Part 2: Architecture & Development Roadmap

```text
Image Digitizer/
├── core/                   # The mathematical heart of the digitizer
│   ├── geometry.py         # Shapely/PyClipper wrappers, unit conversions, offsets
│   ├── pathfinding.py      # Nearest-neighbor / TSP solvers for jump stitch optimization
│   └── stitch_engine.py    # Generators for Running, Concentric, Tatami, and Satin stitches
├── vision/                 # Image processing and vectorization
│   └── vectorizer.py       # OpenCV pipelines for color quantization and contour extraction
├── io/                     # File reading and writing
│   ├── base_format.py      # Abstract base class for embroidery file formats
│   ├── dst_writer.py       # Tajima .DST binary format encoder
│   └── svg_parser.py       # Basic SVG path/polygon importer
├── ui/                     # The frontend and rendering
│   ├── canvas.py           # PyQt6 interactive QGraphicsView stitch viewport
│   └── main_window.py      # Main application window, toolbars, and controls
├── tests/                  # Unit tests for geometry, stitch generation, and binary I/O
├── assets/                 # Icons, cursors, and sample images
├── requirements.txt        # Project dependencies
└── main.py                 # Application entry point
```
