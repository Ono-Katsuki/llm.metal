# LLM Training System - Makefile for Apple Silicon (M4 Pro)

CC       = clang
OBJC     = clang
METAL    = xcrun -sdk macosx metal
METALLIB = xcrun -sdk macosx metallib

# Compiler flags
CFLAGS   = -O3 -march=native -std=c17 -Wall -Wextra -Wno-unused-parameter -DACCELERATE_NEW_LAPACK
OBJCFLAGS = $(CFLAGS) -fobjc-arc
LDFLAGS  = -framework Metal -framework Foundation -framework Accelerate

# Directories
SRC_DIR  = src
BUILD_DIR = build
SHADER_DIR = shaders

# Source files
C_SRCS   = $(SRC_DIR)/core/tensor.c \
           $(SRC_DIR)/core/mem_pool.c \
           $(SRC_DIR)/core/autograd.c \
           $(SRC_DIR)/nn/layers.c \
           $(SRC_DIR)/nn/attention.c \
           $(SRC_DIR)/nn/transformer.c \
           $(SRC_DIR)/nn/optimizer.c \
           $(SRC_DIR)/nn/lora.c \
           $(SRC_DIR)/nn/qwen3.c \
           $(SRC_DIR)/nn/fast_inference.c \
           $(SRC_DIR)/nn/fast_sft.c \
           $(SRC_DIR)/data/tokenizer.c \
           $(SRC_DIR)/data/dataloader.c \
           $(SRC_DIR)/data/gguf.c \
           $(SRC_DIR)/data/safetensors.c \
           $(SRC_DIR)/train/trainer.c

OBJC_SRCS = $(SRC_DIR)/core/metal_backend.m \
           $(SRC_DIR)/nn/fast_metal.m

MAIN_SRC  = main.c

# Object files
C_OBJS    = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(C_SRCS))
OBJC_OBJS = $(patsubst $(SRC_DIR)/%.m,$(BUILD_DIR)/%.o,$(OBJC_SRCS))
MAIN_OBJ  = $(BUILD_DIR)/main.o
ALL_OBJS  = $(C_OBJS) $(OBJC_OBJS) $(MAIN_OBJ)

# Shader files
METAL_SRC = $(SHADER_DIR)/kernels.metal
METAL_AIR = $(BUILD_DIR)/kernels.air
METAL_LIB = $(BUILD_DIR)/kernels.metallib

# Target
TARGET = llm_train

# ===================================================================
# Build rules
# ===================================================================

.PHONY: all clean run test shaders

all: dirs $(TARGET)
	@$(MAKE) shaders 2>/dev/null || echo "Note: Metal shader precompilation skipped (Xcode required). Runtime compilation will be used."

dirs:
	@mkdir -p $(BUILD_DIR)/core $(BUILD_DIR)/nn $(BUILD_DIR)/data $(BUILD_DIR)/train

$(TARGET): $(ALL_OBJS)
	$(CC) $(LDFLAGS) -o $@ $^
	@echo "Build complete: $(TARGET)"

# C source compilation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Objective-C source compilation
$(BUILD_DIR)/core/metal_backend.o: $(SRC_DIR)/core/metal_backend.m
	$(OBJC) $(OBJCFLAGS) -c -o $@ $<

$(BUILD_DIR)/nn/fast_metal.o: $(SRC_DIR)/nn/fast_metal.m
	$(OBJC) $(OBJCFLAGS) -c -o $@ $<

# Main
$(BUILD_DIR)/main.o: main.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Metal shaders
shaders: $(METAL_LIB)

$(METAL_AIR): $(METAL_SRC)
	$(METAL) -c -o $@ $<

$(METAL_LIB): $(METAL_AIR)
	$(METALLIB) -o $@ $<

# ===================================================================
# Run targets
# ===================================================================

run: all
	./$(TARGET) --test --steps 100

# Test with tiny model
test: all
	./$(TARGET) --test --layers 2 --dim 64 --heads 2 --seq 32 --batch 2 --steps 20

# 125M parameter model
run-125M: all
	./$(TARGET) --preset 125M --train data/train.txt --val data/val.txt

# ===================================================================
# Clean
# ===================================================================

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

# ===================================================================
# Dependencies (auto-generated)
# ===================================================================

$(BUILD_DIR)/core/tensor.o: $(SRC_DIR)/core/tensor.h $(SRC_DIR)/core/mem_pool.h
$(BUILD_DIR)/core/mem_pool.o: $(SRC_DIR)/core/mem_pool.h
$(BUILD_DIR)/core/autograd.o: $(SRC_DIR)/core/autograd.h $(SRC_DIR)/core/tensor.h
$(BUILD_DIR)/core/metal_backend.o: $(SRC_DIR)/core/metal_backend.h
$(BUILD_DIR)/nn/layers.o: $(SRC_DIR)/nn/layers.h $(SRC_DIR)/core/tensor.h $(SRC_DIR)/core/autograd.h
$(BUILD_DIR)/nn/attention.o: $(SRC_DIR)/nn/attention.h $(SRC_DIR)/nn/layers.h $(SRC_DIR)/nn/lora.h
$(BUILD_DIR)/nn/transformer.o: $(SRC_DIR)/nn/transformer.h $(SRC_DIR)/nn/attention.h $(SRC_DIR)/nn/layers.h
$(BUILD_DIR)/nn/optimizer.o: $(SRC_DIR)/nn/optimizer.h $(SRC_DIR)/nn/layers.h
$(BUILD_DIR)/nn/lora.o: $(SRC_DIR)/nn/lora.h $(SRC_DIR)/nn/layers.h $(SRC_DIR)/core/tensor.h $(SRC_DIR)/core/autograd.h
$(BUILD_DIR)/nn/qwen3.o: $(SRC_DIR)/nn/qwen3.h $(SRC_DIR)/nn/layers.h $(SRC_DIR)/nn/attention.h $(SRC_DIR)/nn/lora.h $(SRC_DIR)/data/gguf.h $(SRC_DIR)/data/safetensors.h
$(BUILD_DIR)/nn/fast_inference.o: $(SRC_DIR)/nn/fast_inference.h $(SRC_DIR)/nn/qwen3.h $(SRC_DIR)/nn/fast_metal.h
$(BUILD_DIR)/nn/fast_sft.o: $(SRC_DIR)/nn/fast_sft.h $(SRC_DIR)/nn/qwen3.h $(SRC_DIR)/nn/fast_metal.h
$(BUILD_DIR)/nn/fast_metal.o: $(SRC_DIR)/nn/fast_metal.h
$(BUILD_DIR)/data/tokenizer.o: $(SRC_DIR)/data/tokenizer.h
$(BUILD_DIR)/data/dataloader.o: $(SRC_DIR)/data/dataloader.h $(SRC_DIR)/data/tokenizer.h
$(BUILD_DIR)/data/gguf.o: $(SRC_DIR)/data/gguf.h
$(BUILD_DIR)/data/safetensors.o: $(SRC_DIR)/data/safetensors.h
$(BUILD_DIR)/train/trainer.o: $(SRC_DIR)/train/trainer.h $(SRC_DIR)/nn/transformer.h $(SRC_DIR)/nn/optimizer.h
