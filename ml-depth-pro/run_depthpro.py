import depth_pro

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Load and preprocess an image.
# Use the example image from the repo:
image, _, f_px = depth_pro.load_rgb("data/example.jpg")
# If you later want to use your own image, change the path above.
image = transform(image)

# Run inference.
prediction = model.infer(image, f_px=f_px)

# Results
depth = prediction["depth"]                 # Depth in meters
focallength_px = prediction["focallength_px"]  # Focal length in pixels

print("Depth shape:", depth.shape)
print("Focal length (px):", focallength_px)