import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # ======== 解析命令行参数 ========
    parser = argparse.ArgumentParser(description="Story generation with a Hugging Face model")
    parser.add_argument("--model_name", type=str, default="HeAAAAA/story_generation_Llama3.1_8B_SFT", help="Hugging Face model name or path")
    parser.add_argument("--prompt", type=str, default="Once upon a time, in a quiet forest, there was a little fox who dreamed of flying.", help="Prompt text for story generation")
    args = parser.parse_args()

    # ======== 自动选择设备 ========
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Loading model: {args.model_name}")

    # ======== 加载模型和分词器 ========
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # ======== 编码输入并生成 ========
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
    )

    # ======== 解码输出 ========
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n=== Generated Story ===")
    print(story)

if __name__ == "__main__":
    main()
