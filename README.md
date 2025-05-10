data structure and algorithm course opencv lab project 🧪

This project is part of a data structures and algorithms course, focusing on implementing the watershed algorithm using OpenCV. It includes various tasks and solutions related to image segmentation. The primary development environment is Linux, utilizing C/C++ and the OpenCV library.

developed on linux 🐧

* [X] works on linux
* [ ] works on windows

## 🛠️ Installation and Setup

### Prerequisites

* A C++ compiler (e.g., g++) ⚙️
* OpenCV library installed. The following modules are used: 🖼️
  * `opencv_core`
  * `opencv_imgcodecs`
  * `opencv_highgui`
  * `opencv_imgproc`
* `make` utility for building the project. 🧰

### Building the Solutions

The main solutions are located in the `opencv-watershed/solutions/` directory.

1. **Clone the repository:** 📥

   ```bash
   git clone <repository-url>
   cd opencv-course-lab
   ```
2. **Navigate to the solutions directory:** 📁

   ```bash
   cd opencv-watershed/solutions
   ```
3. **Build the tasks:** 🏗️
   Use the provided Makefile to compile the different tasks.

   ```bash
   make
   ```

   This will create executables for `task1`, `task2`, and `task3` inside the `opencv-watershed/solutions/build/` directory.

   To build a specific task (e.g., task1):

   ```bash
   make build/task1
   ```

   To clean the build files: 🧹

   ```bash
   make clean
   ```

## 🎯 Lab Requirements

### Input

- Any M * N color image

### Tools

- VSCode or a compiler on Mac/Linux platform
- OpenCV library
- Pure C language or C/C++ language

### Requirements

- Learn to use the STL Standard Template Library and basic OpenCV data formats
- Minimize the use of static arrays, use pointers and dynamic memory allocation/deallocation
- Reasonably divide the source code into files and functions to ensure module independence
- Follow naming conventions for functions, variables, constants, etc. (avoid using Chinese pinyin)
- Provide macro comments for files and functions, and micro comments for core variables and code segments
- Validate input legality and provide functionality and fault tolerance prompts
- Optimize algorithms as much as possible to ensure stability and low time-space complexity
- Design and optimize the interaction logic and visual UI of the demo 🎨

## ✨ Main Features and Components

This project implements the watershed algorithm and related image processing tasks. The core functionalities are divided into several parts:

### 1. Watershed Segmentation (`opencv-watershed/solutions/task1.cpp`) 🌊

* Performs image segmentation using OpenCV's watershed algorithm.
* **Random Seed Generation**: Automatically generates `K` random seed points for the watershed algorithm. It ensures a minimum distance between seeds, inspired by Poisson disk sampling using a greedy approach. 🎲
* **Interactive Visualization**: Displays the original image, marks the generated seed points with their numbers, and shows the watershed result with semi-transparent, randomly colored segments. 🌈
* **User Controls**: Allows users to input an image, specify the number of seeds (K), and adjust a temperature parameter for seed generation. Key presses control seed generation, visualization, and running the watershed algorithm. ⌨️

### 2. Four-Color Theorem Application (`opencv-watershed/solutions/task2.cpp`) 🎨

* **Adjacency Analysis**: Builds an adjacency list for the regions obtained from the watershed segmentation. 🕸️
* **Graph Coloring**: Applies the four-color theorem to the segmented regions. It attempts to color the graph of regions using a maximum of four colors such that no two adjacent regions share the same color. 🖌️
* **Algorithm**: Uses a backtracking algorithm, potentially with heuristics like coloring the most constrained region first (highest degree in the adjacency graph).
* **Interactive Visualization**: Displays the four-colored segmentation result. Users can mouse over regions to see their IDs. 🖱️

### 3. Region Area Analysis and Huffman Coding (`opencv-watershed/solutions/task3.cpp`) 📊🌳

* **Area Calculation**: Calculates the area (number of pixels) for each segmented region. 📏
* **Heap Sort**: Sorts the regions based on their areas using a heap sort algorithm and displays the minimum and maximum areas. 🔢
* **Range-Based Search**: Allows users to specify an area range (lower and upper bounds). The program then highlights regions whose areas fall within this range. 🔍
* **Huffman Coding**: Uses the areas of the selected (or all) regions as frequencies to build a Huffman tree and generate Huffman codes. 🌲
* **Visualization**: Draws the generated Huffman tree and saves it as a `.dot` file (which can be converted to an image using Graphviz) and a `.png` file. 🖼️

### 4. Utility Functions (`opencv-watershed/solutions/watershed_utils.h`) 🛠️

* Provides common helper functions used across the different tasks:
  * Enhanced command-line interface (CLI) with colored output and icons. 💅
  * User input handling for image paths, K value, temperature, etc.
  * Functions for printing help messages, welcome messages.
  * Image loading and display.
  * Seed generation logic.
  * Area calculation and sorting utilities.

### 5. Original Watershed Demo (`opencv-watershed/watershed/`) 🎞️

* This directory contains an earlier version or a different demonstration of the watershed algorithm, possibly with a frontend component, as indicated by the original `README.md`.

## 🚀 Usage

Clone the repository first if you haven't already:

```bash
git clone <repository-url>
cd opencv-course-lab
```

### Main Solutions (`opencv-watershed/solutions/`)

These tasks are interactive and use keyboard commands to proceed through different stages (e.g., loading image, generating seeds, running algorithms).

1. **Navigate to the solutions directory and build:**

   ```bash
   cd opencv-watershed/solutions
   make
   ```
2. **Run the compiled tasks:** ▶️
   Executables will be in the `opencv-watershed/solutions/build/` directory.

   * **Task 1 (Watershed Segmentation):**

     ```bash
     ./build/task1
     ```

     Follow the on-screen prompts to input the image name and the number of seeds (K). Then use the following hotkeys:

     * `g`: Generate seeds.
     * `v`: Visualize generated seeds on the input image.
     * `w`: Run the watershed algorithm.
     * `r`: Restore the original image and allow re-generation of seeds.
     * `h`: Display help.
     * `q` or `ESC`: Quit.
   * **Task 2 (Four-Coloring):**

     ```bash
     ./build/task2
     ```

     Follow prompts for image and K. Hotkeys include those from Task 1, plus:

     * `c`: Perform four-coloring on the watershed result.
   * **Task 3 (Area Analysis & Huffman Coding):**

     ```bash
     ./build/task3
     ```

     Follow prompts for image and K. Hotkeys include those from Task 1, plus:

     * `s`: Calculate and sort region areas, displaying min and max.
     * `t`: Input an area range to highlight regions and generate a Huffman tree for them.

### Original Watershed Demo with Frontend (`opencv-watershed/watershed/`)

```shell
cd opencv-watershed/watershed
make
cd build
./cv2-watershed
```

This demo might have different usage instructions or a graphical interface not covered by the CLI tasks above.

## 📝 Lab Report Requirement

### Cover Page

- Aesthetic layout including: school and college logo, report title, department, major and class, student ID, name, instructor, submission date, etc.

### Table of Contents

- Detailed to the second-level headings with page numbers

### Problem Description

- Includes: experimental tasks, specification requirements, programming environment, test data, evaluation metrics, etc.

### Algorithm Design

- Includes: problem-solving principle analysis, overall architecture design, algorithm logic design, data structure design, summary of innovative ideas

### Test Analysis

- Includes: regular testing, legality testing, extreme performance testing, comparative testing, test result analysis

### Summary and Outlook

- Includes: summary of experimental gains and problems, directions and ideas for optimization

### Appendix

- Includes: references, core source code with comments 📚

## 🔗 Dependencies and Requirements

* **C++ Compiler**: A modern C++ compiler that supports C++11 or later (e.g., GCC/g++). The project is built using `g++` as specified in the Makefiles. ⚙️
* **OpenCV Library**: Version 4.x is recommended. The following OpenCV modules are essential: 🖼️
  * `opencv_core`: Core functionalities, basic data structures.
  * `opencv_imgcodecs`: Image file reading and writing.
  * `opencv_highgui`: High-level GUI, window management, and user interaction (e.g., `imshow`, `waitKey`).
  * `opencv_imgproc`: Image processing functions, including `watershed`, `cvtColor`, `findContours`, etc.
  * The `LDFLAGS` in the Makefiles (`-lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc`) specify these dependencies.
* **Make**: The `make` utility is used to build the project based on the provided Makefiles. 🧰
* **Graphviz (Optional, for Task 3)**: To visualize the Huffman tree, Graphviz `dot` command-line tool should be installed. Task 3 generates a `.dot` file which can be converted to an image (e.g., PNG) using `dot -Tpng huffman_tree.dot -o huffman_tree.png`. 🌳➡️🖼️
* **Operating System**: Primarily developed and tested on Linux. Windows compatibility is listed as a to-do. 🐧

## 🙌 Contribution Guidelines

Currently, there are no formal contribution guidelines. However, if you wish to contribute, please consider the following:

* **Follow Existing Code Style**: Adhere to the naming conventions and coding style present in the project (e.g., use of STL, dynamic memory, comments). ✍️
* **Modularity**: Keep functions and modules independent and well-documented. 🧱
* **Testing**: Ensure your changes are well-tested and do not break existing functionality. ✅
* **Issue Tracker**: For significant changes or bug fixes, consider opening an issue first to discuss the proposed changes. 🐛
* **Pull Requests**: Submit changes via pull requests with clear descriptions of the modifications. ➡️📦

This project adheres to general academic integrity principles. Ensure any contributions are your own work or properly attributed. 🧑‍🎓
