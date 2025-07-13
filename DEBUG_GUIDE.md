# DSnoT Custom Debugging Guide

## Overview
This enhanced debugging setup allows you to start debugging from specific points in your code, rather than from the beginning every time.

## New Custom Debug Points Available

1. **start** - Beginning of main function
2. **model_load** - Right before model loading
3. **tokenizer** - Right before tokenizer loading  
4. **pruning** - Right before pruning operations
5. **evaluation** - Right before evaluation
6. **save** - Right before model saving

## How to Use Custom Debug Points

### Method 1: Interactive Debug Script (Recommended)
```bash
python custom_debug.py
```
This will show you a menu to choose exactly where to start debugging:
```
DSnoT Custom Debug Entry Points
========================================
Choose where to start debugging:
1. Start of main function
2. Model loading
3. Tokenizer loading
4. Pruning operation
5. Evaluation
6. Model saving
0. Run without debugging

Enter your choice (0-6):
```

### Method 2: Direct Command Line
```bash
# Debug from pruning operation
python main.py --model babylm/babyllama-10m-2024 --prune_method DSnoT --initial_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --max_cycle_time 50 --update_threshold 0.1 --pow_of_var_regrowing 1 --debug --debug_point pruning

# Debug from model loading
python main.py --model babylm/babyllama-10m-2024 --prune_method DSnoT --initial_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --max_cycle_time 50 --update_threshold 0.1 --pow_of_var_regrowing 1 --debug --debug_point model_load
```

### Method 3: Data Loading Specific Debug
```bash
python debug_data.py
```
This provides targeted debugging for data loading issues:
```
Data Loading Debug Options
==============================
1. Debug C4 data loading (get_c4 function)
2. Debug tokenization process
3. Debug SparseGPT data loading (get_loaders)
```

## Example Usage Scenarios

### Scenario 1: Model Loading Issues
```bash
python custom_debug.py
# Choose option 2 (Model loading)
# Debugger will start right before model loading
```

### Scenario 2: Tokenization/Data Issues  
```bash
python debug_data.py
# Choose option 2 (Debug tokenization process)
# Debugger will start in tokenization function
```

### Scenario 3: Pruning Algorithm Issues
```bash
python custom_debug.py
# Choose option 4 (Pruning operation)
# Debugger will start right before pruning
```

## Quick Debug Commands Reference

When debugger starts:
- `l` - Show current code location
- `n` - Next line
- `s` - Step into functions
- `c` - Continue execution
- `p variable_name` - Print variable value
- `pp variable_name` - Pretty print variable
- `w` - Show where you are in the call stack
- `u` - Go up one level in stack
- `d` - Go down one level in stack
- `q` - Quit debugger
