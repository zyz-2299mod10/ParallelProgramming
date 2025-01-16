# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++20 -Wall -O3 -fopenmp -Iinclude #-g -Iinclude

# Source files and targets
SRC1 = one_level.cpp
SRC2 = two_level.cpp
TARGET1 = Puzzle_solver_1
TARGET2 = Puzzle_solver_2

# Default rule to build all targets
all: $(TARGET1) $(TARGET2)

# Rule to build Puzzle_solver_1
$(TARGET1): $(SRC1)
	$(CXX) $(CXXFLAGS) -o $(TARGET1) $(SRC1)

# Rule to build Puzzle_solver_2
$(TARGET2): $(SRC2)
	$(CXX) $(CXXFLAGS) -o $(TARGET2) $(SRC2)

# Clean rule to remove generated files
clean:
	rm -f $(TARGET1) $(TARGET2)

# Phony targets
.PHONY: all clean
