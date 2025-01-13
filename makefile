NVCC = nvcc
CXXFLAGS = -std=c++17 -O2
LDFLAGS = -lcudart_static

# Directories
SRC_DIR = src
BUILD_DIR = build

# Output binary
OUTPUT = program

# Source files
CU_SRCS = $(wildcard $(SRC_DIR)/*.cu)
CPP_SRCS = $(wildcard $(SRC_DIR)/*.cpp)
ALL_SRCS = $(CU_SRCS) $(CPP_SRCS)
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.obj, $(patsubst $(SRC_DIR)/%.cu, $(BUILD_DIR)/%.obj, $(ALL_SRCS)))

# Default target
all: $(OUTPUT)

# Link all objects into the executable
$(OUTPUT): $(OBJS) | $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	rm $(OUTPUT).lib
	rm $(OUTPUT).exp

# Compile CUDA and C++ sources
$(BUILD_DIR)/%.obj: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.obj: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) -x cu -c $< -o $@

# Ensure the build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm $(OUTPUT).exe

run:
	@$(OUTPUT) 

.PHONY: all clean
