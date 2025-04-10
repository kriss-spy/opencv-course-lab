#include <cmath>
#include <iostream>

int computeHexagonalGridPoints(int M, int N, double d)
{
    // For a regular hexagonal grid:
    // - Horizontal spacing: d
    // - Vertical spacing: d * sqrt(3)/2

    const double verticalSpacing = d * std::sqrt(3.0) / 2.0;

    // Calculate number of points in each dimension
    // Add 1 to include potential partial rows/columns
    int numRows = static_cast<int>(M / verticalSpacing) + 1;
    int numColsOdd = static_cast<int>(N / d) + 1;
    int numColsEven = static_cast<int>(N / d) + 1;

    // In a hexagonal grid, even and odd rows may have different numbers of points
    int totalPoints = (numRows / 2) * numColsEven + ((numRows + 1) / 2) * numColsOdd;

    return totalPoints;
}

int main()
{
    int M = 600;     // Image height
    int N = 600;     // Image width
    double d = 60.0; // Distance between points

    int points = computeHexagonalGridPoints(M, N, d);
    std::cout << "Number of points in hexagonal grid: " << points << std::endl;

    return 0;
}