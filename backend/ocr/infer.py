import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
MODEL_ID = "deepseek-ai/DeepSeek-OCR"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

model = AutoModel.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map=None,                # IMPORTANT
    attn_implementation="eager",
    low_cpu_mem_usage=True
)

# 🔥 FORCE GPU
model = model.cuda().eval()

print("Model device:", next(model.parameters()).device)
def run_ocr(image_path):
    import os # Import os for directory operations
    
    # DeepSeek-OCR model requires an output path, even if temporary
    output_dir = "deepseek_ocr_temp_output"
    os.makedirs(output_dir, exist_ok=True)

    prompt = "<image>\nFree OCR."

    with torch.inference_mode():
        # The model's infer method expects output_path and save_results=True
        # to produce result files that can be read.
        model.infer(
            tokenizer,
            prompt=prompt,
            image_file=image_path,
            output_path=output_dir, # Provide an output path
            base_size=768,
            image_size=512,
            crop_mode=False,
            save_results=True # Set to True to ensure results are saved
        )
    
    # Read the OCR result from the generated files
    extracted_text = ""
    result_file_mmd = os.path.join(output_dir, "result.mmd")
    result_file_txt = os.path.join(output_dir, "result.txt")
    
    if os.path.exists(result_file_mmd):
        with open(result_file_mmd, "r", encoding="utf-8") as f:
            extracted_text = f.read().strip()
    elif os.path.exists(result_file_txt):
        with open(result_file_txt, "r", encoding="utf-8") as f:
            extracted_text = f.read().strip()
            
    # Clean up the temporary output directory
    # import shutil # Uncomment if you want to clean up the directory
    # shutil.rmtree(output_dir) # Uncomment if you want to clean up the directory

    return extracted_text

print(run_ocr("/home/kosik/Documents/hvac-ocr-module/test_images/sample_nameplate.png"))