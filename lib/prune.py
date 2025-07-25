import time

import torch
import torch.nn as nn
import transformers

from .save_results import save_time_result
from .data import get_loaders
from .sparsegpt import SparseGPT
from .layerwrapper import WrappedGPT


def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count) / total_params


def prepare_calibration_input(args, model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=device,
    )
    inps.requires_grad = False
    cache = {"i": 0, "attention_mask": None, "position_ids": None, "cache_position": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            cache["position_ids"] = kwargs.get("position_ids", None)
            cache["cache_position"] = kwargs.get("cache_position", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    cache_position = cache["cache_position"]
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids, cache_position


def return_reorder_indice(input_tensor):
    """
    For instance:
    [[1., -2., 3.],
    [-2, 2., -4],
    [5., 6., -7],
    [-6, -7, -4]]
    return indices of
    [[-2.,  3.,  1.],
    [-2., -4.,  2.],
    [-7.,  6.,  5.],
    [-6., -7., -4.]]
    Description: The relative order in the positive number remains unchanged, and the relative order in the negative number is flipped.
    """
    positive_tensor = input_tensor.clone()
    negative_tensor = input_tensor.clone()

    positive_mask = positive_tensor > 0
    negative_mask = negative_tensor < 0

    positive_indices = (
        torch.arange(0, input_tensor.shape[1], device=input_tensor.device)
        .to(torch.float64)
        .repeat(input_tensor.shape[0], 1)
    )
    negative_indices = (
        torch.arange(0, input_tensor.shape[1], device=input_tensor.device)
        .to(torch.float64)
        .repeat(input_tensor.shape[0], 1)
    )

    positive_indices[~positive_mask] = float("inf")
    negative_indices[~negative_mask] = float("inf")

    positive_value, _ = torch.sort(positive_indices, dim=1)
    negative_value, _ = torch.sort(negative_indices, dim=1)

    positive_value = torch.flip(positive_value, dims=[1])

    negative_value[negative_value == float("inf")] = 0
    positive_value[positive_value == float("inf")] = 0

    reorder_indice = (positive_value + negative_value).to(torch.int64)

    return reorder_indice

def prune_magnitude(
    args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0
):
    layers = model.model.layers

    total_time = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            start_time = time.time()
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = torch.zeros_like(W) == 1
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][
                    int(W.numel() * args.sparsity_ratio)
                ].cpu()
                W_mask = W_metric <= thresh

            W[W_mask] = 0
            end_time = time.time()
            total_time += end_time - start_time

    if args.get_time_overhead:
        save_time_result(args, args.output_results_file, total_time)


def prune_wanda(
    args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, save_path=None
):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    
    num_params_before = sum(p.numel() for p in model.parameters()) / 1e6

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        "c4",
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids, cache_position = prepare_calibration_input(
            args, model, dataloader, device
        )

    total_time = 0
    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if (
            f"model.layers.{i}" in model.hf_device_map
        ):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids, cache_position = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev) if attention_mask is not None else None,
                position_ids.to(dev) if position_ids is not None else None,
                cache_position.to(dev) if cache_position is not None else None,
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                layer_kwargs = {
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                }
                if cache_position is not None:
                    layer_kwargs["cache_position"] = cache_position
                
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    **layer_kwargs
                )[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            start_time = time.time()
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1))
            )

            W_mask = (
                torch.zeros_like(W_metric) == 1
            )  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii : (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                # unstructured pruning
                indices = sort_res[1][:, : int(W_metric.shape[1] * args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero
            end_time = time.time()
            total_time += end_time - start_time

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        inps, outs = outs, inps

    if args.get_time_overhead:
        save_time_result(args, args.output_results_file, total_time)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


     # ================ ADDED THIS CODE FOR SAVING THE PRUNED MODEL USING WANDA METHOD=================
     # Clean up memory
    model.eval()
        # Calculate final parameters
    num_params_after = sum(p.numel() for p in model.parameters()) / 1e6
    compression_ratio = (num_params_before - num_params_after) / num_params_before * 100
    
    # print(f"Pruned model parameters: {num_params_after:.2f}M")
    print("Validating parameter count after full pruning and reconfiguration...")
    print(f"✅ Final Pruned Model Parameters: {num_params_after:.2f}M")

    print(f"Compression ratio: {compression_ratio:.2f}%")

    # Save the pruned model if save_path is provided
    saved_path = None
    if save_path is not None:
        print(f"Saving pruned model to {save_path}...")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        saved_path = save_path
        print(f"Pruned model saved successfully to {save_path}")



@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0, save_path=None):
    print("Starting pruning process...")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token.")

    num_params_before = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Original model parameters: {num_params_before:.2f}M")

    print("Loading dataset...")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer
    )
    print("Dataset loaded.")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    print(f"Total layers to prune: {len(layers)}")

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]
        print(f"Device set from hf_device_map: {dev}")

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    print("Initialized input tensor.")

    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            print(f"Catcher forward pass: sample {cache['i']}")
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError  # used for control flow

    layers[0] = Catcher(layers[0])
    print("Catcher injected into first layer.")

    print("Feeding data to model...")
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    print("Data fed. Removing Catcher.")
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    print("Cache collected. Starting pruning.")

    total_time = 0

    for i in range(len(layers)):
        print(f"\n--- Pruning Layer {i} ---")
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"Layer {i} device: {dev}")
            inps, outs, attention_mask, position_ids = (
                inps.to(dev),
                outs.to(dev),
                attention_mask.to(dev),
                position_ids.to(dev),
            )

        print("Finding prunable sublayers...")
        subset = find_layers(layer)
        print(f"Found {len(subset)} sublayers to prune.")

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])
            print(f"Initialized SparseGPT for {name}")

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            print(f"Registering forward hook for {name}")
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        print("Running forward passes for calibration...")
        for j in range(args.nsamples):
            print(f"Layer {i}, Sample {j}")
            layer_kwargs = {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
            # Note: sparsegpt uses older format, no cache_position needed here
            outs[j] = layer(
                inps[j].unsqueeze(0),
                **layer_kwargs
            )[0]

        print("Removing forward hooks.")
        for h in handles:
            h.remove()

        for name in gpts:
            print(f"Pruning sublayer {name} in layer {i}")
            start_time = time.time()

            gpts[name].fasterprune(
                args.sparsity_ratio,
                prune_n=prune_n,
                prune_m=prune_m,
                percdamp=0.01,
                blocksize=128,
            )

            end_time = time.time()
            prune_time = end_time - start_time
            total_time += prune_time
            print(f"Pruned {name} in {prune_time:.2f}s")

            gpts[name].free()

        print(f"Validating pruned layer {i} with forward pass...")
        for j in range(args.nsamples):
            layer_kwargs = {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
            # Note: sparsegpt uses older format, no cache_position needed here
            outs[j] = layer(
                inps[j].unsqueeze(0),
                **layer_kwargs
            )[0]

        layers[i] = layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        print(f"Finished Layer {i}")

    print("All layers pruned.")

    if args.get_time_overhead:
        print(f"Saving pruning time overhead: {total_time:.2f}s")
        save_time_result(args, args.output_results_file, total_time)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print("Pruning complete --- Saving pruned model...")



    # ================ ADDED THIS CODE FOR SAVING THE PRUNED MODEL USING SPARSE GPT=================
     # Clean up memory
    model.eval()
        # Calculate final parameters
    num_params_after = sum(p.numel() for p in model.parameters()) / 1e6
    compression_ratio = (num_params_before - num_params_after) / num_params_before * 100
    
    # print(f"Pruned model parameters: {num_params_after:.2f}M")
    print("Validating parameter count after full pruning and reconfiguration...")
    print(f"✅ Final Pruned Model Parameters: {num_params_after:.2f}M")

    print(f"Compression ratio: {compression_ratio:.2f}%")

    # Save the pruned model if save_path is provided
    saved_path = None
    if save_path is not None:
        print(f"Saving pruned model to {save_path}...")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        saved_path = save_path
        print(f"Pruned model saved successfully to {save_path}")


def prune_DSnoT(
    args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, save_path=None
):
    print("Starting DSNOT PRUNING....")
    use_cache = model.config.use_cache
    model.config.use_cache = False

    model.eval()
    num_params_before = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Original model parameters: {num_params_before:.2f}M")
    model.train()

    print("loading calibdation data")
    dataloader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed, seqlen=2048, tokenizer=tokenizer
    )
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids, cache_position = prepare_calibration_input(
            args, model, dataloader, device
        )

    total_time = 0
    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if (f"model.layers.{i}" in model.hf_device_map):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            
            print("This was the condition of LLama-30B and Llama-65B and is now removed...")
            # dev = model.hf_device_map[f"model.layers.{i}"]
            # inps, outs, attention_mask, position_ids = (
            #     inps.to(dev),
            #     outs.to(dev),
            #     attention_mask.to(dev),
            #     position_ids.to(dev),
            # )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(
                subset[name],
                initial_method=args.initial_method
            )

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # the below code is used to get the calibration data for the DSnoT pruning
        print("Running forward passes for calibration...")
        for j in range(args.nsamples):
            with torch.no_grad():
                layer_kwargs = {
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                }
                if cache_position is not None:
                    layer_kwargs["cache_position"] = cache_position
                
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    **layer_kwargs
                )[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            start_time = time.time()

            DSnoT_metric = subset[name].weight.data * wrapped_layers[name].sum_metric_row.reshape((1, -1))

            if args.initial_method == "wanda":
                initial_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                    wrapped_layers[name].scaler_row.reshape((1, -1))
                )
            elif args.initial_method == "magnitude":
                initial_metric = torch.abs(subset[name].weight.data)
            elif args.initial_method == "sparsegpt":
                W = subset[name].weight.data.clone()
                if isinstance(subset[name], nn.Conv2d):
                    W = W.flatten(1)
                if isinstance(subset[name], transformers.Conv1D):
                    W = W.t()
                W = W.float()

                H = wrapped_layers[name].H
                # del wrapped_layers[name].H
                dead = torch.diag(H) == 0
                H[dead, dead] = 1
                W[:, dead] = 0

                percdamp = 0.01
                damp = percdamp * torch.mean(torch.diag(H))
                diag = torch.arange(
                    wrapped_layers[name].columns, device=wrapped_layers[name].dev
                )
                H[diag, diag] += damp
                H = torch.linalg.cholesky(H)
                H = torch.cholesky_inverse(H)
                H = torch.linalg.cholesky(H, upper=True)
                Hinv = H

                initial_metric = W**2 / (torch.diag(Hinv).reshape((1, -1))) ** 2

            weight_mask = torch.zeros_like(initial_metric) == 1

            if prune_n != 0:
                if (name.split(".")[0] == args.skip_layer or name.split(".")[1] == args.skip_sub_layer):
                    for ii in range(initial_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = initial_metric[:, ii : (ii + prune_m)].float()
                            weight_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True,)
                else:
                    initial_prune_indices = torch.zeros((initial_metric.shape[0], 0), dtype=torch.int64, device=initial_metric.device,)
                    initial_res_indices = torch.zeros((initial_metric.shape[0], 0), dtype=torch.int64, device=initial_metric.device,)

                    for ii in range(initial_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = initial_metric[:, ii : (ii + prune_m)].float()
                            _, tmp_all_indices = torch.sort(tmp, dim=1)
                            tmp_all_indices += ii
                            res_prune_n = prune_m - prune_n
                            tmp_indices, tmp_res_indices = torch.split(
                                tmp_all_indices,
                                split_size_or_sections=[prune_n, res_prune_n],
                                dim=1,
                            )

                            initial_prune_indices = torch.cat(
                                (initial_prune_indices, tmp_indices), dim=1
                            )
                            initial_res_indices = torch.cat(
                                (initial_res_indices, tmp_res_indices), dim=1
                            )
                            weight_mask.scatter_(1, tmp_indices, True)

                    metric_for_regrowing = DSnoT_metric.clone()

                    metric_for_regrowing.scatter_(1, initial_res_indices, 0)

                    reconstruction_error = torch.sum(metric_for_regrowing, dim=1, keepdim=True)
                    initialize_error_sign = torch.sign(reconstruction_error)

                    if args.pow_of_var_regrowing:
                        metric_for_regrowing /= torch.pow(
                            wrapped_layers[name].var.reshape((1, -1)),
                            args.pow_of_var_regrowing,
                        )

                    _, regrowing_indices_block = torch.sort(metric_for_regrowing, dim=1, stable=True)

                    indice_indice_list_for_regrowing = torch.zeros(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    last_one = regrowing_indices_block.shape[-1] - 1
                    indice_indice_list_for_regrowing[:, 1] = last_one
                    update_num_for_regrowing = torch.ones(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    update_num_for_regrowing[:, 1] = -1

                    initial_metric.scatter_(1, initial_prune_indices, float("inf"))
                    W_metric_max_value = (torch.max(initial_metric, dim=1, keepdim=True)[0] + 1)

                    cycle_time = 1
                    update_mask = torch.ones_like(
                        reconstruction_error, dtype=torch.bool
                    )
                    while not (
                        torch.all(update_mask == False)
                        or cycle_time > args.max_cycle_time
                    ):
                        cycle_time += 1

                        # regrowing
                        indice_of_indice_indice_list_for_regrowing = (
                            (reconstruction_error > 0).int().to(torch.int64)
                        )
                        indice_indice_for_regrowing = torch.gather(
                            indice_indice_list_for_regrowing,
                            1,
                            indice_of_indice_indice_list_for_regrowing,
                        )

                        regrowing_indice = torch.gather(
                            regrowing_indices_block,
                            1,
                            indice_indice_for_regrowing.to(torch.int64),
                        )

                        regrowing_metric = DSnoT_metric.gather(
                            1, regrowing_indice.to(torch.int64)
                        )

                        recover_block_start_indice = (
                            regrowing_indice - regrowing_indice % prune_m
                        )

                        recover_block_indices = (
                            torch.arange(
                                0, prune_m, device=recover_block_start_indice.device
                            ).repeat(recover_block_start_indice.shape[1], 1)
                            + recover_block_start_indice
                        )

                        pruning_block = torch.gather(
                            initial_metric, 1, recover_block_indices.to(torch.int64)
                        )

                        pruning_wanda_metric, pruning_indice = torch.topk(
                            pruning_block, 1, dim=1, largest=False
                        )

                        pruning_indice += recover_block_start_indice

                        
                        pruning_metric = DSnoT_metric.gather( 1, pruning_indice.to(torch.int64) )
                        

                        reconstruction_error_after = ( reconstruction_error + pruning_metric - regrowing_metric )

                        update_mask = (update_mask & ( initialize_error_sign == torch.sign(reconstruction_error_after) ) & ( abs(reconstruction_error) > args.update_threshold))

                        initial_metric.scatter_(1, pruning_indice, W_metric_max_value)

                        weight_mask.scatter_(1, pruning_indice, update_mask)

                        weight_mask.scatter_(1, regrowing_indice, ~update_mask)

                        reconstruction_error += torch.where(
                            update_mask,
                            pruning_metric,
                            torch.zeros_like(pruning_metric),
                        )
                        reconstruction_error -= torch.where(
                            update_mask,
                            regrowing_metric,
                            torch.zeros_like(regrowing_metric),
                        )

                        indice_indice_list_for_regrowing.scatter_(
                            1,
                            indice_of_indice_indice_list_for_regrowing,
                            indice_indice_for_regrowing
                            + update_num_for_regrowing.gather(
                                1, indice_of_indice_indice_list_for_regrowing
                            ),
                        )
            else:
                _, sorted_initial_indice = torch.sort(
                    initial_metric, dim=-1, stable=True
                )

                sparsity_num = int(initial_metric.shape[1] * args.sparsity_ratio)
                res_sparsity_num = sorted_initial_indice.shape[1] - sparsity_num

                initial_prune_indices, initial_res_indices = torch.split(
                    sorted_initial_indice,
                    split_size_or_sections=[sparsity_num, res_sparsity_num],
                    dim=1,
                )

                if (
                    name.split(".")[0] == args.skip_layer
                    or name.split(".")[1] == args.skip_sub_layer
                    or args.without_DSnoT
                ):
                    weight_mask.scatter_(1, initial_prune_indices, True)

                else:
                    weight_mask.scatter_(1, initial_prune_indices, True)

                    metric_for_regrowing = DSnoT_metric.clone()
                    wanda_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                        wrapped_layers[name].scaler_row.reshape((1, -1))
                    )

                    metric_for_regrowing.scatter_(1, initial_res_indices, 0)
                    reconstruction_error = torch.sum(
                        metric_for_regrowing, dim=1, keepdim=True
                    )
                    initialize_error_sign = torch.sign(reconstruction_error)

                    if args.pow_of_var_regrowing:
                        metric_for_regrowing /= torch.pow(
                            wrapped_layers[name].var.reshape((1, -1)),
                            args.pow_of_var_regrowing,
                        )

                    _, regrowing_indices_block = torch.sort(
                        metric_for_regrowing, dim=1, stable=True
                    )

                    wanda_metric.scatter_(1, initial_prune_indices, float("inf"))
                    wanda_res_indices, _ = torch.split(
                        torch.sort(wanda_metric, dim=1, stable=True)[1],
                        split_size_or_sections=[res_sparsity_num, sparsity_num],
                        dim=1,
                    )
                    reorder_indice_of_pruning_indice = return_reorder_indice(
                        torch.gather(DSnoT_metric, 1, wanda_res_indices)
                    )
                    pruning_indices_block = torch.gather(
                        wanda_res_indices, 1, reorder_indice_of_pruning_indice
                    )

                    indice_indice_list_for_regrowing = torch.zeros(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    last_one = regrowing_indices_block.shape[-1] - 1
                    indice_indice_list_for_regrowing[:, 1] = last_one

                    update_num_for_regrowing = torch.ones(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    update_num_for_regrowing[:, 1] = -1

                    indice_indice_list_for_pruning = torch.zeros(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    last_one = pruning_indices_block.shape[-1] - 1
                    indice_indice_list_for_pruning[:, 1] = last_one

                    update_num_for_pruning = torch.ones(
                        (reconstruction_error.shape[0], 2),
                        device=reconstruction_error.device,
                        dtype=torch.long,
                    )
                    update_num_for_pruning[:, 1] = -1

                    update_mask = torch.ones_like(
                        reconstruction_error, dtype=torch.bool
                    )
                    cycle_time = 0
                    while not ( torch.all(update_mask == False) or cycle_time >= args.max_cycle_time ):
                        cycle_time += 1
                        
                        # regrowing
                        indice_of_indice_indice_list_for_regrowing = (
                            (reconstruction_error > 0).int().to(torch.int64)
                        )

                        indice_indice_for_regrowing = torch.gather(
                            indice_indice_list_for_regrowing,
                            1,
                            indice_of_indice_indice_list_for_regrowing,
                        )

                        regrowing_indice = torch.gather(
                            regrowing_indices_block,
                            1,
                            indice_indice_for_regrowing.to(torch.int64),
                        )

                        regrowing_metric = DSnoT_metric.gather(
                            1, regrowing_indice.to(torch.int64)
                        )

                        indice_indice_list_for_regrowing.scatter_(
                            1,
                            indice_of_indice_indice_list_for_regrowing,
                            indice_indice_for_regrowing
                            + update_num_for_regrowing.gather(
                                1, indice_of_indice_indice_list_for_regrowing
                            ),
                        )

                        # pruning
                        indice_of_indice_indice_list_for_pruning = (
                            (reconstruction_error < 0).int().to(torch.int64)
                        )

                        indice_indice_for_pruning = torch.gather(
                            indice_indice_list_for_pruning,
                            1,
                            indice_of_indice_indice_list_for_pruning,
                        )

                        pruning_indice = torch.gather(
                            pruning_indices_block,
                            1,
                            indice_indice_for_pruning.to(torch.int64),
                        )

                        pruning_metric = DSnoT_metric.gather(
                            1, pruning_indice.to(torch.int64)
                        )

                        indice_indice_list_for_pruning.scatter_(
                            1,
                            indice_of_indice_indice_list_for_pruning, 
                            indice_indice_for_pruning
                            + update_num_for_pruning.gather(
                                1, indice_of_indice_indice_list_for_pruning
                            ),
                        )

                        # change mask
                        reconstruction_error_after = (
                            reconstruction_error + pruning_metric - regrowing_metric
                        )

                        if args.without_same_sign == str(True):
                            update_mask = update_mask & (
                                abs(reconstruction_error) > args.update_threshold
                            )
                        else:
                            update_mask = (
                                update_mask
                                & (abs(reconstruction_error) > args.update_threshold)
                                & (
                                    initialize_error_sign
                                    == torch.sign(reconstruction_error_after)
                                )
                            )

                        weight_mask.scatter_(1, pruning_indice, update_mask)
                        weight_mask.scatter_(1, regrowing_indice, ~update_mask)

                        reconstruction_error += torch.where(
                            update_mask,
                            pruning_metric,
                            torch.zeros_like(pruning_metric),
                        )
                        reconstruction_error -= torch.where(
                            update_mask,
                            regrowing_metric,
                            torch.zeros_like(regrowing_metric),
                        )

            
            subset[name].weight.data[weight_mask] = 0

            end_time = time.time()
            total_time += end_time - start_time

        for j in range(args.nsamples):
            with torch.no_grad():
                layer_kwargs = {
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                }
                if cache_position is not None:
                    layer_kwargs["cache_position"] = cache_position
                
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    **layer_kwargs
                )[0]
        inps, outs = outs, inps

    if args.get_time_overhead:
        save_time_result(args, args.output_results_file, total_time)

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


    # ================ ADDED THIS CODE FOR SAVING THE PRUNED MODEL USING DSNOT GPT=================
    model.eval()
      # Clean up memory

        # Calculate final parameters
    num_params_after = sum(p.numel() for p in model.parameters()) / 1e6
    compression_ratio = (num_params_before - num_params_after) / num_params_before * 100
    
    # print(f"Pruned model parameters: {num_params_after:.2f}M")
    print("Validating parameter count after full pruning and reconfiguration...")
    print(f"✅ Final Pruned Model Parameters: {num_params_after:.2f}M")

    print(f"Compression ratio: {compression_ratio:.2f}%")

    # Save the pruned model if save_path is provided
    saved_path = None
    if save_path is not None:
        print(f"Saving pruned model to {save_path}...")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        saved_path = save_path
        print(f"Pruned model saved successfully to {save_path}")





