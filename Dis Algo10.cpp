#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>

void printResults(const std::string& name, double time) {
    std::cout << name << " : " << time << "s\n";
}

// Bubble Sort
void bubbleSort(std::vector<int>& data) {
    bool swapped; // Flag to indicate if a swap occurred
    for (size_t i = 0; i < data.size() - 1; i++) { // Outer loop to control the number of passes
        swapped = false; // Reset the swapped flag for each pass
        for (size_t j = 0; j < data.size() - i - 1; j++) { // Inner loop for comparing adjacent elements
            if (data[j] > data[j + 1]) { // If the current element is greater than the next element
                std::swap(data[j], data[j + 1]); // Swap the elements
                swapped = true; // Set the swapped flag to true
            }
        }
        if (!swapped) // If no swaps occurred, the array is sorted
            break; // Exit the loop
    }
}

// Bubble Sort Parallel
void bubbleSortParallel(std::vector<int>& data) {
    size_t n = data.size(); // Get the size of the array
    for (size_t i = 0; i < n - 1; ++i) { // Outer loop to control the number of passes
        bool swapped = false; // Flag to indicate if a swap occurred
#pragma omp parallel for shared(swapped) // Parallelize the inner loop
        for (size_t j = 0; j < n - i - 1; ++j) { // Inner loop for comparing adjacent elements
            if (data[j] > data[j + 1]) { // If the current element is greater than the next element
                std::swap(data[j], data[j + 1]); // Swap the elements
                swapped = true; // Set the swapped flag to true
            }
        }
        if (!swapped) // If no swaps occurred, the array is sorted
            break; // Exit the loop
    }
}

// Insertion Sort
void insertionSort(std::vector<int>& data) {
    int key, j; // Variables to store the key element and the index for comparison
    for (size_t i = 1; i < data.size(); i++) { // Loop through elements starting from the second element
        key = data[i]; // Set the key to the current element
        j = i - 1; // Initialize the comparison index to the previous element
        while (j >= 0 && data[j] > key) { // While the comparison index is valid and the key is smaller than the compared element
            data[j + 1] = data[j]; // Shift the compared element to the right
            j = j - 1; // Move the comparison index to the left
        }
        data[j + 1] = key; // Insert the key in the correct position
    }
}

// Insertion Sort Parallel
void insertionSortParallel(std::vector<int>& data) {
    size_t n = data.size(); // Get the size of the array
#pragma omp parallel for // Parallelize the outer loop
    for (size_t i = 1; i < n; ++i) { // Loop through elements starting from the second element
        int key = data[i]; // Set the key to the current element
        int j = i - 1; // Initialize the comparison index to the previous element
        while (j >= 0 && data[j] > key) { // While the comparison index is valid and the key is smaller than the compared element
            data[j + 1] = data[j]; // Shift the compared element to the right
            --j; // Move the comparison index to the left
        }
        data[j + 1] = key; // Insert the key in the correct position
    }
}

// Selection Sort
void selectionSort(std::vector<int>& data) {
    size_t min_idx; // Variable to store the index of the minimum element
    for (size_t i = 0; i < data.size() - 1; i++) { // Loop through the array
        min_idx = i; // Assume the current element is the minimum
        for (size_t j = i + 1; j < data.size(); j++) { // Loop to find the minimum element in the remaining array
            if (data[j] < data[min_idx]) // If a smaller element is found
                min_idx = j; // Update the index of the minimum element
        }
        std::swap(data[i], data[min_idx]); // Swap the minimum element with the current element
    }
}

// Selection Sort Parallel
void selectionSortParallel(std::vector<int>& data) {
    size_t n = data.size(); // Get the size of the array
#pragma omp parallel for // Parallelize the outer loop
    for (size_t i = 0; i < n - 1; ++i) { // Loop through the array
        size_t min_idx = i; // Assume the current element is the minimum
        for (size_t j = i + 1; j < n; ++j) { // Loop to find the minimum element in the remaining array
            if (data[j] < data[min_idx]) // If a smaller element is found
                min_idx = j; // Update the index of the minimum element
        }
        if (min_idx != i) // If the minimum element is not the current element
            std::swap(data[i], data[min_idx]); // Swap the minimum element with the current element
    }
}

// Merge Sort Main
void merge(std::vector<int>& data, int left, int mid, int right) {
    std::vector<int> temp(right - left + 1); // Temporary vector to store the merged elements
    int i = left, j = mid + 1, k = 0; // Initialize indices for the left, right, and temporary vectors
    while (i <= mid && j <= right) { // While both halves have elements to compare
        if (data[i] <= data[j]) { // If the element in the left half is smaller or equal
            temp[k++] = data[i++]; // Add it to the temporary vector and move the left index
        }
        else { // If the element in the right half is smaller
            temp[k++] = data[j++]; // Add it to the temporary vector and move the right index
        }
    }
    while (i <= mid) { // If there are remaining elements in the left half
        temp[k++] = data[i++]; // Add them to the temporary vector
    }
    while (j <= right) { // If there are remaining elements in the right half
        temp[k++] = data[j++]; // Add them to the temporary vector
    }
    for (i = left, k = 0; i <= right; i++, k++) { // Copy the merged elements back to the original array
        data[i] = temp[k];
    }
}

// Merge Sort Serial
void mergeSortSerial(std::vector<int>& data, int left, int right) {
    if (left < right) { // If there is more than one element
        int mid = left + (right - left) / 2; // Find the middle point
        mergeSortSerial(data, left, mid); // Recursively sort the left half
        mergeSortSerial(data, mid + 1, right); // Recursively sort the right half
        merge(data, left, mid, right); // Merge the two halves
    }
}
void mergeSortSerial(std::vector<int>& data) {
    mergeSortSerial(data, 0, data.size() - 1); // Call the recursive merge sort function
}

// Merge Sort Parallel
void mergeSortParallel(std::vector<int>& data, int left, int right) {
    if (left < right) { // If there is more than one element
        int mid = left + (right - left) / 2; // Find the middle point
#pragma omp parallel sections // Parallelize the sorting of the two halves
        {
#pragma omp section
            mergeSortParallel(data, left, mid); // Sort the left half
#pragma omp section
            mergeSortParallel(data, mid + 1, right); // Sort the right half
        }
        merge(data, left, mid, right); // Merge the two halves
    }
}
void mergeSortParallel(std::vector<int>& data) {
    mergeSortParallel(data, 0, data.size() - 1); // Call the recursive merge sort function
}

// Quick Sort Main 
int partition(std::vector<int>& data, int low, int high) {
    int pivot = data[high]; // Choose the last element as the pivot
    int i = (low - 1); // Index of the smaller element
    for (int j = low; j <= high - 1; j++) { // Loop through the array
        if (data[j] < pivot) { // If the current element is smaller than the pivot
            i++; // Increment the index of the smaller element
            std::swap(data[i], data[j]); // Swap the current element with the element at the smaller index
        }
    }
    std::swap(data[i + 1], data[high]); // Swap the pivot with the element at the smaller index + 1
    return (i + 1); // Return the partitioning index
}

// Quick Sort Serial
void quickSortSerial(std::vector<int>& data, int low, int high) {
    if (low < high) { // If there are more than one elements
        int pi = partition(data, low, high); // Partition the array and get the partitioning index
        quickSortSerial(data, low, pi - 1); // Recursively sort the left half
        quickSortSerial(data, pi + 1, high); // Recursively sort the right half
    }
}
void quickSortSerial(std::vector<int>& data) {
    quickSortSerial(data, 0, data.size() - 1); // Call the recursive quick sort function
}

// Quick Sort Parallel
void quickSortParallel(std::vector<int>& data, int low, int high) {
    if (low < high) { // If there are more than one elements
        int pi = partition(data, low, high); // Partition the array and get the partitioning index
#pragma omp parallel sections // Parallelize the sorting of the two halves
        {
#pragma omp section
            quickSortParallel(data, low, pi - 1); // Sort the left half
#pragma omp section
            quickSortParallel(data, pi + 1, high); // Sort the right half
        }
    }
}
void quickSortParallel(std::vector<int>& data) {
    quickSortParallel(data, 0, data.size() - 1); // Call the recursive quick sort function
}

// Heap Sort Main
void heapify(std::vector<int>& data, int n, int i) {
    int largest = i; // Initialize the largest as root
    int left = 2 * i + 1; // Left child
    int right = 2 * i + 2; // Right child

    if (left < n && data[left] > data[largest]) // If left child is larger than root
        largest = left; // Update the largest

    if (right < n && data[right] > data[largest]) // If right child is larger than the largest so far
        largest = right; // Update the largest

    if (largest != i) { // If the largest is not root
        std::swap(data[i], data[largest]); // Swap root and the largest
        heapify(data, n, largest); // Recursively heapify the affected sub-tree
    }
}

// Heap Sort Serial
void heapSortSerial(std::vector<int>& data) {
    int n = data.size(); // Get the size of the array
    for (int i = n / 2 - 1; i >= 0; i--) // Build heap (rearrange array)
        heapify(data, n, i);

    for (int i = n - 1; i >= 0; i--) { // One by one extract an element from heap
        std::swap(data[0], data[i]); // Move current root to end
        heapify(data, i, 0); // Call heapify on the reduced heap
    }
}

// Heap Sort Parallel
void heapSortParallel(std::vector<int>& data) {
    int n = data.size(); // Get the size of the array
#pragma omp parallel for // Parallelize the heap building
    for (int i = n / 2 - 1; i >= 0; --i)
        heapify(data, n, i);

#pragma omp parallel for // Parallelize the heap extraction
    for (int i = n - 1; i >= 0; --i) {
        std::swap(data[0], data[i]); // Move current root to end
        heapify(data, i, 0); // Call heapify on the reduced heap
    }
}


// Shell Sort Serial
void shellSortSerial(std::vector<int>& data) {
    for (int gap = data.size() / 2; gap > 0; gap /= 2) { // Start with a large gap and reduce it
        for (size_t i = gap; i < data.size(); i++) { // Perform gapped insertion sort
            int temp = data[i]; // Store the current element
            size_t j;
            for (j = i; j >= gap && data[j - gap] > temp; j -= gap) // Shift earlier gap-sorted elements up
                data[j] = data[j - gap];
            data[j] = temp; // Place the current element in its correct position
        }
    }
}

// Shell Sort Parallel
void shellSortParallel(std::vector<int>& data) {
    size_t n = data.size(); // Get the size of the array
    for (int gap = n / 2; gap > 0; gap /= 2) { // Start with a large gap and reduce it
#pragma omp parallel for // Parallelize the gapped insertion sort
        for (size_t i = gap; i < n; ++i) {
            int temp = data[i]; // Store the current element
            size_t j;
            for (j = i; j >= gap && data[j - gap] > temp; j -= gap) // Shift earlier gap-sorted elements up
                data[j] = data[j - gap];
            data[j] = temp; // Place the current element in its correct position
        }
    }
}

// Radix Sort Main
int getMax(const std::vector<int>& data) {
    return *std::max_element(data.begin(), data.end()); // Get the maximum element in the array
}

void countSortRadix(std::vector<int>& data, int exp) {
    std::vector<int> output(data.size()); // Output array to store the sorted elements
    int count[10] = { 0 }; // Count array to store the frequency of digits

    for (size_t i = 0; i < data.size(); i++) // Count the occurrences of each digit
        count[(data[i] / exp) % 10]++;

    for (int i = 1; i < 10; i++) // Change count[i] so that count[i] contains the actual position of the digit
        count[i] += count[i - 1];

    for (int i = data.size() - 1; i >= 0; i--) { // Build the output array
        output[count[(data[i] / exp) % 10] - 1] = data[i];
        count[(data[i] / exp) % 10]--;
    }

    for (size_t i = 0; i < data.size(); i++) // Copy the output array to data
        data[i] = output[i];
}

// Radix Sort Serial
void radixSortSerial(std::vector<int>& data) {
    int max = getMax(data); // Get the maximum element

    for (int exp = 1; max / exp > 0; exp *= 10) // Apply counting sort for every digit
        countSortRadix(data, exp);
}

// Radix Sort Parallel
void radixSortParallel(std::vector<int>& data) {
    int max = getMax(data); // Get the maximum element

#pragma omp parallel for // Parallelize the counting sort for each digit
    for (int exp = 1; max / exp > 0; exp *= 10)
        countSortRadix(data, exp);
}

// Counting Sort Serial
void countingSortSerial(std::vector<int>& data) {
    int max = *std::max_element(data.begin(), data.end()); // Get the maximum element
    int min = *std::min_element(data.begin(), data.end()); // Get the minimum element
    int range = max - min + 1; // Calculate the range of the elements

    std::vector<int> count(range), output(data.size()); // Count array to store the frequency of elements and output array

    for (size_t i = 0; i < data.size(); i++) // Count the occurrences of each element
        count[data[i] - min]++;

    for (size_t i = 1; i < range; i++) // Change count[i] so that count[i] contains the actual position of the element
        count[i] += count[i - 1];

    for (int i = data.size() - 1; i >= 0; i--) { // Build the output array
        output[count[data[i] - min] - 1] = data[i];
        count[data[i] - min]--;
    }

    for (size_t i = 0; i < data.size(); i++) // Copy the output array to data
        data[i] = output[i];
}

// Counting Sort Parallel
void countingSortParallel(std::vector<int>& data) {
    int max = *std::max_element(data.begin(), data.end()); // Get the maximum element
    int min = *std::min_element(data.begin(), data.end()); // Get the minimum element
    int range = max - min + 1; // Calculate the range of the elements

    std::vector<int> count(range), output(data.size()); // Count array to store the frequency of elements and output array

#pragma omp parallel for // Parallelize the counting of occurrences
    for (size_t i = 0; i < data.size(); i++)
        count[data[i] - min]++;

#pragma omp parallel for // Parallelize the cumulative addition
    for (size_t i = 1; i < range; i++)
        count[i] += count[i - 1];

#pragma omp parallel for // Parallelize the building of the output array
    for (int i = data.size() - 1; i >= 0; i--) {
        output[count[data[i] - min] - 1] = data[i];
        count[data[i] - min]--;
    }

#pragma omp parallel for // Parallelize the copying of the output array to data
    for (size_t i = 0; i < data.size(); i++)
        data[i] = output[i];
}

// Bucket Sort Serial
void bucketSortSerial(std::vector<int>& data) {
    int bucketCount = sqrt(data.size()); // Determine the number of buckets
    std::vector<std::vector<int>> buckets(bucketCount); // Create the buckets

    int max = *std::max_element(data.begin(), data.end()); // Get the maximum element
    int min = *std::min_element(data.begin(), data.end()); // Get the minimum element
    int range = max - min + 1; // Calculate the range of the elements

    for (size_t i = 0; i < data.size(); i++) { // Distribute the elements into buckets
        int bucketIndex = ((data[i] - min) * bucketCount) / range;
        buckets[bucketIndex].push_back(data[i]);
    }

    for (size_t i = 0; i < bucketCount; i++) // Sort each bucket
        std::sort(buckets[i].begin(), buckets[i].end());

    size_t index = 0; // Merge the buckets back into the data array
    for (size_t i = 0; i < bucketCount; i++) {
        for (size_t j = 0; j < buckets[i].size(); j++) {
            data[index++] = buckets[i][j];
        }
    }
}

// Bucket Sort Parallel
void bucketSortParallel(std::vector<int>& data) {
    int bucketCount = sqrt(data.size()); // Determine the number of buckets
    std::vector<std::vector<int>> buckets(bucketCount); // Create the buckets

    int max = *std::max_element(data.begin(), data.end()); // Get the maximum element
    int min = *std::min_element(data.begin(), data.end()); // Get the minimum element
    int range = max - min + 1; // Calculate the range of the elements
#pragma omp parallel
    {
        std::vector<int> local_bucket; // Local bucket for each thread
#pragma omp for nowait
        for (size_t i = 0; i < data.size(); ++i) { // Distribute the elements into buckets
            int bucketIndex = ((data[i] - min) * bucketCount) / range;
#pragma omp critical
            buckets[bucketIndex].push_back(data[i]);
        }
    }
#pragma omp parallel for // Sort each bucket in parallel
    for (size_t i = 0; i < bucketCount; ++i)
        std::sort(buckets[i].begin(), buckets[i].end());

    size_t index = 0; // Merge the buckets back into the data array
    for (size_t i = 0; i < bucketCount; ++i) {
        for (size_t j = 0; j < buckets[i].size(); ++j) {
            data[index++] = buckets[i][j];
        }
    }
}

// Generate Random Data
std::vector<int> generateRandomData(size_t size) {
    std::vector<int> data(size); // Create a vector of the specified size
    std::generate(data.begin(), data.end(), std::rand); // Fill the vector with random numbers
    return data; // Return the vector
}


// Main Function
int main() {
    size_t dataSize; // Declare variable to store the size of the data array
    std::cout << "Enter data size: "; // Prompt the user to enter the size of the data
    std::cin >> dataSize; // Read the input size into dataSize

    // Variables to measure time duration of sorting algorithms
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    double time_taken;

    // Array of algorithm names for display purposes
    std::string algorithms[10] = {
        "Bubble Sort", "Insertion Sort", "Selection Sort",
        "Merge Sort", "Quick Sort", "Heap Sort",
        "Shell Sort", "Radix Sort", "Counting Sort", "Bucket Sort"
    };

    // Array of function pointers for serial sorting functions
    void (*serialFuncs[10])(std::vector<int>&) = {
        bubbleSort, insertionSort, selectionSort,
        mergeSortSerial, quickSortSerial, heapSortSerial,
        shellSortSerial, radixSortSerial, countingSortSerial, bucketSortSerial
    };

    // Array of function pointers for parallel sorting functions
    void (*parallelFuncs[10])(std::vector<int>&) = {
        bubbleSortParallel, insertionSortParallel, selectionSortParallel,
        mergeSortParallel, quickSortParallel, heapSortParallel,
        shellSortParallel, radixSortParallel, countingSortParallel, bucketSortParallel
    };

    omp_set_num_threads(omp_get_max_threads()); // Set the number of threads to the maximum available

    for (int i = 0; i < 10; i++) {
        // Generate new data for each sorting algorithm to ensure fair comparison
        std::vector<int> data = generateRandomData(dataSize);

        // Use a fresh copy of the newly generated unsorted data for serial sorting
        std::vector<int> copyData = data;
        start = std::chrono::high_resolution_clock::now(); // Start timer
        serialFuncs[i](copyData); // Sort using the serial function
        end = std::chrono::high_resolution_clock::now(); // End timer
        time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0; // Calculate time taken
        printResults(algorithms[i] + " Serial", time_taken); // Print the results for serial sorting

        // Generate new data again for parallel sorting to ensure data is unsorted
        data = generateRandomData(dataSize);
        copyData = data;
        start = std::chrono::high_resolution_clock::now(); // Start timer
        parallelFuncs[i](copyData); // Sort using the parallel function
        end = std::chrono::high_resolution_clock::now(); // End timer
        time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0; // Calculate time taken
        printResults(algorithms[i] + " Parallel", time_taken); // Print the results for parallel sorting
    }

    return 0; // End of main function
}


