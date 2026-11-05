"""Check which timm model names are actually available."""

import timm

print("--- efficientnetv2 model names ---")
for name in timm.list_models("*efficientnetv2*"):
    print(name)

print("\n--- with pretrained weights ---")
for name in timm.list_pretrained("*efficientnetv2*"):
    print(name)

print("\n--- tf_efficientnetv2 variants ---")
for name in timm.list_models("*tf_efficientnetv2*"):
    print(name)
