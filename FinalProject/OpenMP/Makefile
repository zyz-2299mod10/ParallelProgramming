# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++20 -Wall -O3 -Iinclude -fopenmp #-g -Iinclude

# Find all source files automatically
SRCDIR = .
OBJDIR = obj
Debeg = $(shell find $(SRCDIR) -name "*.txt")
SRCS = $(shell find $(SRCDIR) -name "*.cpp")
OBJS = $(patsubst %.cpp, $(OBJDIR)/%.o, $(SRCS))

# Target executable
TARGET = Puzzle_solver

# Default rule to build the target
all: $(TARGET)

# Rule to build the target executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Rule to compile source files into object files
$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)   # Create the necessary directory for object files
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule to remove generated files
clean:
	rm -rf $(OBJDIR) $(TARGET) $(Debeg)

# Phony targets
.PHONY: all clean
