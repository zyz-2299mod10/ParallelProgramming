# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++20 -Wall -O3 -g -Iinclude -mavx -msse4.2 -march=native#-g -Iinclude

# Find all source files automatically
SRCDIR = ./src/
OBJDIR = obj
SRCS = main_thread_no_mutex.cpp
SRCS += $(shell find $(SRCDIR) -name "*.cpp")
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
	rm -rf $(OBJDIR) $(TARGET)

# Phony targets
.PHONY: all clean
