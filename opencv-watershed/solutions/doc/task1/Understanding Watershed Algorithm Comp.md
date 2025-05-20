# Understanding Watershed Algorithm Components

In the watershed algorithm implementation, there are several key components used to segment an image into regions:

## 1. Seeds
- **What they are**: Points that serve as the starting locations for each region in the segmentation
- **In the code**: Stored as `vector<Point> seeds` 
- **How they're used**: 
  - Generated with `generate_seeds()` function
  - Act as markers for where different regions will grow from
  - Visualized in the image with `visualize_seeds()`

## 2. Marker Mask
- **What it is**: A binary or labeled image that identifies the initial regions
- **In the code**: Stored as `Mat marker_mask` with type CV_32SC1 (32-bit signed integers)
- **How it's used**:
  - Initially created as a grayscale version of the input image
  - Seeds are placed onto this mask
  - Contours are found in this mask to initialize distinct watershed regions

## 3. Markers
- **What they are**: A labeled image where each pixel is assigned a region ID
- **In the code**: Stored as `Mat markers` with type CV_32SC1
- **How they're used**:
  - Initialized with zeros, then populated with region IDs (1, 2, 3, etc.) based on contours
  - The watershed algorithm updates this matrix during execution
  - After watershed completion:
    - Regions are labeled with positive values (1, 2, 3, etc.)
    - Boundaries between regions are marked with -1
    - Background or unlabeled areas are 0

## 4. Contours
- **What they are**: Boundaries that define the initial regions
- **In the code**: Stored as `vector<vector<Point>> contours`
- **How they're used**:
  - Found in the marker mask using `findContours()`
  - Each contour becomes a uniquely labeled region in the markers matrix
  - Each region is assigned a unique color for visualization

The watershed algorithm then simulates water rising from these seed regions, creating watersheds (boundaries) where different region waters would meet, resulting in a segmentation of the image.