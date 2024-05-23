Uses slerp to extract a single expert from multiple experts in an MoE model. This process requires a significant amount of system memory and can take some time. Be patient and make sure you have about five times the system RAM as the model's parameter count from which you're extracting.
-Lucas Atkins (Crystalcareai)

## Example usage:
```python extract.py --model-name mistralai/Mixtral-8x7B-v0.1 --output-dir ./out```

## Note
Only works for models using Mixture architecture. There is no guarantee that we'll expand this to support other architectures in the future.

