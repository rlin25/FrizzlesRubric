import os
from safetensors import safe_open

def test_model():
    model_path = "models/grammar_model/model.safetensors"
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    
    try:
        # Try to open the safetensors file
        with safe_open(model_path, framework="pt") as f:
            # Get the tensor names
            tensor_names = f.keys()
            print(f"Successfully loaded model with tensors: {list(tensor_names)}")
            return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

if __name__ == "__main__":
    test_model() 