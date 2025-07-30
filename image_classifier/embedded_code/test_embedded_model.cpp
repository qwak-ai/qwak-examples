// test_embedded_model.cpp
#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: test_model <model_file>\n";
        return 1;
    }

    std::string model_file = argv[1];
    std::ifstream file(model_file, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.close();

    if (size < 1024) { // Example check: Model should be at least 1KB
        std::cerr << "Model file too small. Test failed.\n";
        return 1;
    }

    std::cout << "Model file size: " << size << " bytes\n";
    std::cout << "TEST_PASSED\n"; // Changed output for easier parsing
    return 0;
}