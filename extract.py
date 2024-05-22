import argparse
import json
import os
import torch
from safetensors.torch import safe_open, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def slerp(v0, v1, lambda_val):
    v0_norm = torch.linalg.norm(v0, dim=-1, keepdims=True)
    v1_norm = torch.linalg.norm(v1, dim=-1, keepdims=True)
    v0_normalized = v0 / v0_norm
    v1_normalized = v1 / v1_norm
    dot = torch.sum(v0_normalized * v1_normalized, dim=-1, keepdims=True)
    dot = torch.clamp(dot, -1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    # Handling the zero sin_omega case element-wise
    sin_omega_zero_mask = sin_omega == 0
    scale_0 = torch.sin((1 - lambda_val) * omega) / sin_omega
    scale_1 = torch.sin(lambda_val * omega) / sin_omega
    # Prevent division by zero by setting scales to 0 where sin_omega is zero
    scale_0 = torch.where(sin_omega_zero_mask, torch.zeros_like(scale_0), scale_0)
    scale_1 = torch.where(sin_omega_zero_mask, torch.zeros_like(scale_1), scale_1)
    slerped = scale_0 * v0 + scale_1 * v1
    # Linear interpolation where sin_omega is zero
    linear_interpolated = (1 - lambda_val) * v0 + lambda_val * v1
    result = torch.where(sin_omega_zero_mask, linear_interpolated, slerped)
    return result

def interpolate_model(model_name, output_dir, lambda_val=0.7):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    num_layers = model.config.num_hidden_layers

    # Update the configuration
    model.config.num_experts_per_tok = 1
    model.config.num_local_experts = 1

    weight_map_file = os.path.join(model_name, "model.safetensors.index.json")
    with open(weight_map_file, "r") as f:
        weight_map_data = json.load(f)
    weight_map = weight_map_data["weight_map"]

    print("Starting model interpolation...")
    averaged_weights = {}
    for layer_i in range(num_layers):
        print(f"Processing layer {layer_i}...")

        # Process gate weights using slerp
        gate_weight_name = f"model.layers.{layer_i}.block_sparse_moe.gate.weight"
        gate_weight_file = os.path.join(model_name, weight_map[gate_weight_name])
        with safe_open(gate_weight_file, framework="pt") as f:
            gate_weights = f.get_tensor(gate_weight_name)
        slerped_gate_weight = gate_weights[:, 0]
        for i in range(1, 8):
            slerped_gate_weight = slerp(slerped_gate_weight, gate_weights[:, i], lambda_val)
        averaged_weights[gate_weight_name] = slerped_gate_weight.unsqueeze(1)

        # Process expert weights using slerp
        for weight_type in ["w1", "w2", "w3"]:
            expert_weights = []
            for expert_i in range(8):
                print(f"  Processing expert {expert_i}/7")
                weight_name = f"model.layers.{layer_i}.block_sparse_moe.experts.{expert_i}.{weight_type}.weight"
                weight_file = os.path.join(model_name, weight_map[weight_name])
                with safe_open(weight_file, framework="pt") as f:
                    weight = f.get_tensor(weight_name)
                expert_weights.append(weight)

            # Apply slerp to combine the weights
            slerped_weight = expert_weights[0]
            for next_weight in expert_weights[1:]:
                slerped_weight = slerp(slerped_weight, next_weight, lambda_val)
            averaged_weights[f"model.layers.{layer_i}.block_sparse_moe.experts.0.{weight_type}.weight"] = slerped_weight

    # Process non-expert and non-gate weights
    for weight_name, weight_file in weight_map.items():
        if "block_sparse_moe" not in weight_name:
            with safe_open(os.path.join(model_name, weight_file), framework="pt") as f:
                weight = f.get_tensor(weight_name)
            averaged_weights[weight_name] = weight

    print("Model interpolation completed.")

    # Save the interpolated weights
    averaged_weight_map = {}
    for weight_name, weight_tensor in averaged_weights.items():
        shard_file_name = "model.safetensors"
        averaged_weight_map[weight_name] = shard_file_name

    shard_file_path = os.path.join(output_dir, "model.safetensors")
    save_file(averaged_weights, shard_file_path)
    print(f"Interpolated weights saved to {shard_file_path}")

    # Save the weight map to a JSON file
    weight_map_file = os.path.join(output_dir, "model.safetensors.index.json")
    with open(weight_map_file, "w") as f:
        json.dump({
            "metadata": {
                "total_size": sum(tensor.numel() * tensor.element_size() for tensor in averaged_weights.values()),
                "format": "pt",
                "pytorch_version": torch.__version__,
            },
            "weight_map": averaged_weight_map
        }, f)
    print(f"Weight map saved to {weight_map_file}")

    # Save the modified configuration and tokenizer
    model.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Modified configuration and tokenizer saved to {output_dir}")

    # Get a list of all safetensor files in the output directory
    safetensor_files = [file for file in os.listdir(output_dir) if file.endswith(".safetensors")]

    for safetensor_file in safetensor_files:
        safetensors_path = os.path.join(output_dir, safetensor_file)
        tensors = dict()
        try:
            # Open the safetensors file in read mode
            with safe_open(safetensors_path, framework="pt") as f:
                # Iterate over all keys in the safetensors file
                for key in f.keys():
                    # Load each tensor using its key and store it in the 'tensors' dictionary
                    tensors[key] = f.get_tensor(key)
            # Save the tensors back to the safetensors file with added metadata
            save_file(tensors, safetensors_path, metadata={'format': 'pt'})
            print(f"Tensors in {safetensor_file} have been successfully saved with metadata.")
        except Exception as e:
            print(f"An error occurred for {safetensor_file}: {str(e)}")

    # Load the model from the safetensors file
    model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        config=model.config,
        ignore_mismatched_sizes=True,
        torch_dtype="auto",
    )

    # Save the model, configuration, and tokenizer to the output directory
    model.save_pretrained(output_dir, max_shard_size="10GB")
    model.config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model, configuration, and tokenizer saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, help="Name or path of the model repository on the Hugging Face Hub")
    parser.add_argument("--output-dir", required=True, help="Location to write the interpolated HF model")
    parser.add_argument("--lambda-val", type=float, default=0.5, help="Interpolation coefficient")
    args = parser.parse_args()
    interpolate_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        lambda_val=args.lambda_val,
    )

if __name__ == "__main__":
    main()
