

import time

import torch
import torch.nn as nn
import transformers


def prune_model(
        model, tokenizer, 
        pruning_type="magnitude",
        weight_metric="l2", model_type="llama", 
        device=torch.device("cuda:0"),
        max_seq_len=2048, pruning_ratio=0.2, 
            ):
    """
    Prune a language model using Torch Pruning https://github.com/VainF/Torch-Pruning/tree/master.
    
    Args:
        model (hf - model)
        pruning_type (str): Type of pruning importance measure. Options:
                           - "magnitude": GroupMagnitudeImportance (default)
                           - "taylor": TaylorImportance  (NOT IMPLEMENTED YET)
                           - "hessian": HessianImportance (NOT IMPLEMENTED YET)
                           - "random": RandomImportance (NOT IMPLEMENTED YET)
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model
        device (torch.device): Device to run the model on (default: "cuda:0")
        weight_metric (str): Metric for weight importance ("l2", "l1", etc.)
        model_type (str): Type of model architecture ("llama", "gpt2", "bert", etc.")
        max_seq_len (int): Maximum sequence length for the model
        pruning_ratio (float): Ratio of parameters to prune (0.0 to 1.0)
    
    Returns:
        dict: Dictionary containing model statistics and paths
    """

  
    

    # ONLY LLAMA MODEL SUPPORTED FOR NOW
    if model_type.lower() != "llama":
        raise ValueError("Currently only 'llama' model type is supported for pruning.")

    
    if not isinstance(pruning_ratio, float) or not (0.0 < pruning_ratio < 1.0):
        raise ValueError("pruning_ratio must be a float between 0.0 and 1.0.")
    

    if not isinstance(max_seq_len, int) or max_seq_len <= 0:
        raise ValueError("max_seq_len must be a positive integer.")
    
    if not isinstance(pruning_type, str):
        raise ValueError("pruning_type must be a string.")
    
    if not isinstance(weight_metric, str):
        raise ValueError("weight_metric must be a string.")

    model.eval()
    
    # Calculate original parameters
    num_params_before = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Original model parameters: {num_params_before:.2f}M")
    
    
   
    # Print model stats before pruning
    tp.utils.print_tool.before_pruning(model)
    
    # Prepare dummy input for dependency graph construction
    text = "Hello world."
    inputs = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(device)
    
    # Analyze model structure to find attention and MLP layers
    num_heads = {}
    out_channel_groups = {}
    seperate_qkv = False
    
    for name, m in model.named_modules():
        if name.endswith("self_attn"):
            if hasattr(m, "q_proj"):
                seperate_qkv = True
                num_heads[m.q_proj] = model.config.num_attention_heads
                num_heads[m.k_proj] = model.config.num_key_value_heads
                num_heads[m.v_proj] = model.config.num_key_value_heads
            elif hasattr(m, "qkv_proj"):
                seperate_qkv = False
                num_heads[m.qkv_proj] = model.config.num_attention_heads
        if name.endswith('mlp'):
            if hasattr(m, "gate_up_proj"):
                out_channel_groups[m.gate_up_proj] = 2
    
    # Set pruning configuration based on type
    _is_gqa = model.config.num_attention_heads != model.config.num_key_value_heads
    head_pruning_ratio = pruning_ratio
    hidden_size_pruning_ratio = pruning_ratio
    



    if weight_metric == "l2":
      w_metric = 2
    elif weight_metric == "l1":
      w_metric = 1


    # Select importance measure based on pruning_type
    if pruning_type.lower() == "magnitude":
        importance = tp.importance.GroupMagnitudeImportance(p=w_metric, group_reduction='mean')
    elif pruning_type.lower() == "taylor":
        print("NOT IMPLEMENTED YET!! \n using magnitude L2 importance instead")
        importance = tp.importance.GroupMagnitudeImportance(p=2, group_reduction='mean')
        # importance = tp.importance.TaylorImportance()
    elif pruning_type.lower() == "hessian":
        importance = tp.importance.GroupMagnitudeImportance(p=2, group_reduction='mean')
        print("NOT IMPLEMENTED YET!! \n using magnitude L2 importance instead")
        # importance = tp.importance.HessianImportance()
    elif pruning_type.lower() == "random":
        importance = tp.importance.GroupMagnitudeImportance(p=2, group_reduction='mean')
        print("NOT IMPLEMENTED YET!! \n using magnitude L2 importance instead  ")
        # importance = tp.importance.RandomImportance()
    else:
        print(f"Warning: Unknown pruning type '{pruning_type}'. Using magnitude l2 importance.")
        importance = tp.importance.GroupMagnitudeImportance(p=2, group_reduction='mean')
   
    # Create a pruner with specified configuration
    pruner = tp.pruner.BasePruner(
        model, 
        example_inputs=inputs,
        importance=importance,
        global_pruning=False,
        output_transform=lambda x: x.logits,
        pruning_ratio=hidden_size_pruning_ratio,
        ignored_layers=[model.lm_head],
        num_heads=num_heads,
        prune_num_heads=True,
        prune_head_dims=False,
        head_pruning_ratio=head_pruning_ratio,
        out_channel_groups=out_channel_groups,
        round_to=4,
    )
    
    # Execute pruning steps interactively
    print("Starting pruning process...")
    for g in pruner.step(interactive=True):
        g.prune()
    
    # Update model attributes after pruning
    model.config.hidden_size = model.lm_head.in_features
    for name, m in model.named_modules():
        if name.endswith("self_attn"):
            if seperate_qkv:
                m.hidden_size = m.q_proj.out_features
            else:
                m.hidden_size = m.qkv_proj.out_features // 3        
            m.num_heads = m.hidden_size // m.head_dim
            model.config.num_attention_heads = m.num_heads
            if not _is_gqa:
                m.num_key_value_heads = m.num_heads
                model.config.num_key_value_heads = m.num_heads
            if hasattr(m, "num_key_value_groups"):
                m.num_key_value_groups = m.num_heads // model.config.num_key_value_heads
        elif name.endswith("mlp"):
            if hasattr(m, "gate_proj"):
                m.hidden_size = m.gate_proj.in_features
                model.config.intermediate_size = m.gate_proj.out_features
            elif hasattr(m, "gate_up_proj"):
                m.hidden_size = m.gate_up_proj.in_features
                model.config.intermediate_size = m.gate_up_proj.out_features // 2
            else:
                raise ValueError("Unknown mlp layer")
    
    # Final GQA config update
    if not _is_gqa:
        model.config.num_key_value_heads = model.config.num_attention_heads
    
    # Print model stats after pruning
    tp.utils.print_tool.after_pruning(model, do_print=True)
    
    # Clean up memory
    del pruner
    torch.cuda.empty_cache()
    model.eval()
    
    # Calculate final parameters
    num_params_after = sum(p.numel() for p in model.parameters()) / 1e6
    compression_ratio = (num_params_before - num_params_after) / num_params_before * 100
    
    print(f"Pruned model parameters: {num_params_after:.2f}M")
    print(f"Compression ratio: {compression_ratio:.2f}%")

    
    # Return statistics
    return {
        "original_params": num_params_before,
        "pruned_params": num_params_after,
        "compression_ratio": compression_ratio,
        "pruning_type": pruning_type,
        "pruning_ratio": pruning_ratio,
        "model_config": model.config
    }
