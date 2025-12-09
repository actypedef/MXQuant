import torch
import argparse
import sys
from typing import Optional
from model_utils import reorder_model_llama, reorder_model_qwen, reorder_model_mixtral

def get_llama(model_path):
    """加载Llama模型"""
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    from transformers import LlamaForCausalLM
    # 使用 AutoTokenizer 来自动加载正确的 tokenizer 类型
    from transformers import AutoTokenizer
    
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    # 关键修改：使用 AutoTokenizer，并明确关闭 legacy 模式
    tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
    
    model.seqlen = 2048
    return model, tokenizer

def get_qwen(model_path):
    """加载Qwen模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def get_mixtral(model_path):
    """加载Mixtral模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def load_quantized_model(model_path, model_name, kv_cache=False, device='cuda:0'):
    """加载并量化模型"""
    # 确定模型类型
    if "llama" in model_path.lower():
        model, tokenizer = get_llama(model_path)
        reorder_model_func = reorder_model_llama
    elif "qwen" in model_path.lower():
        model, tokenizer = get_qwen(model_path)
        reorder_model_func = reorder_model_qwen
    elif "mixtral" in model_path.lower():
        model, tokenizer = get_mixtral(model_path)
        reorder_model_func = reorder_model_mixtral
    else:
        raise ValueError(f"Unsupported model type: {model_path}")
    
    model.eval()
    
    # 加载量化的索引文件
    import os
    index_filename = f'./saved/{model_name}_reorder_index_wikitext2_mean.pt'
    p6_num_filename = f'./saved/{model_name}_p6_num_wikitext2_mean.pt'
    p8_num_filename = f'./saved/{model_name}_p8_num_wikitext2_mean.pt'
    
    if not os.path.isfile(index_filename):
        # 尝试其他可能的文件名
        index_filename = f'./saved/{model_name}_reorder_index_wikitext2_hessian.pt'
        p6_num_filename = f'./saved/{model_name}_p6_num_wikitext2_hessian.pt'
        p8_num_filename = f'./saved/{model_name}_p8_num_wikitext2_hessian.pt'
        
    if not os.path.isfile(index_filename):
        raise FileNotFoundError(f"Cannot find reorder index file for {model_name}. "
                              f"Expected at: {index_filename}")
    
    print("Loading cached reordering index from disk...")
    reorder_index = torch.load(index_filename, weights_only=False)
    p6_nums = torch.load(p6_num_filename, weights_only=False)
    p8_nums = torch.load(p8_num_filename, weights_only=False)
    
    print("Reordering and quantizing model...")
    model = reorder_model_func(
        model, device=device, kv_cache=kv_cache, 
        reorder_index=reorder_index, p8_nums=p8_nums, p6_nums=p6_nums
    )
    
    # 将模型移到GPU
    model.to(device)
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, device='cuda:0', 
                      max_new_tokens=200, temperature=0.7, top_p=0.9):
    """生成回复（使用正确的对话格式）"""
    
    # 1. 构建模型期待的对话消息格式
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 2. 使用tokenizer的内置模板将消息格式化为模型所需的输入文本
    # 这是最关键的一步，它会自动添加 <|begin_of_text|>、<|start_header_id|>user<|end_header_id|> 等特殊token
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # 先不tokenize，为了查看格式，但这里直接返回文本
        add_generation_prompt=True  # 添加提示让模型开始生成
    )
    
    # 3. 将格式化后的文本转换为模型输入
    input_ids = tokenizer(input_text, return_tensors="pt").to(device)
    
    # 4. 生成参数设置：使用 max_new_tokens 控制生成长度
    generation_config = {
        "max_new_tokens": max_new_tokens,  # 控制新生成token的数量，避免过长
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    # 5. 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **input_ids,
            **generation_config
        )
    
    # 6. 解码输出（跳过输入部分）
    input_length = input_ids.input_ids.shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    return response

def interactive_mode(model, tokenizer, device='cuda:0'):
    """交互模式"""
    print("\n" + "="*50)
    print("模型加载完成！输入你的问题（输入 'quit' 或 'exit' 退出）")
    print("="*50 + "\n")
    
    while True:
        try:
            # 获取用户输入
            prompt = input(">>> ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            
            if not prompt:
                continue
            
            print("\n思考中...", end="", flush=True)
            
            # 生成回复
            response = generate_response(
                model, tokenizer, prompt, device,
                max_new_tokens=256, temperature=0.7, top_p=0.9  # 将 max_length 改为 max_new_tokens
            )
            
            print("\n" + "-"*50)
            print("回复:")
            print(response)
            print("-"*50 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n检测到中断，正在退出...")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            continue

def single_prompt_mode(model, tokenizer, prompt, device='cuda:0'):
    """单次提示模式"""
    print(f"输入: {prompt}")
    print("\n生成回复中...")
    
    response = generate_response(
        model, tokenizer, prompt, device,
        max_length=512, temperature=0.7, top_p=0.9
    )
    
    print("\n" + "="*50)
    print("回复:")
    print(response)
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="量化模型交互式Demo")
    
    # 必需参数
    parser.add_argument(
        'model_path',
        type=str,
        help='模型路径（HuggingFace格式的checkpoint）'
    )
    
    # 可选参数
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='直接提供提示词（如果不提供则进入交互模式）'
    )
    parser.add_argument(
        '--kv_cache',
        action='store_true',
        help='是否量化KV缓存'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='设备（如 cuda:0, cuda:1, cpu）'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='生成的最大长度'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='生成温度（越高越随机）'
    )
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='top-p采样参数'
    )
    
    args = parser.parse_args()
    
    try:
        # 从模型路径提取模型名称
        model_name = args.model_path.split('/')[-2] if '/' in args.model_path else args.model_path
        
        print(f"加载模型: {args.model_path}")
        print(f"模型名称: {model_name}")
        print(f"设备: {args.device}")
        print(f"量化KV缓存: {args.kv_cache}")
        
        # 加载量化模型
        model, tokenizer = load_quantized_model(
            args.model_path, 
            model_name, 
            kv_cache=args.kv_cache,
            device=args.device
        )
        
        print("模型加载完成！")
        
        # 根据是否有prompt选择模式
        if args.prompt:
            single_prompt_mode(
                model, tokenizer, args.prompt, args.device
            )
        else:
            interactive_mode(model, tokenizer, args.device)
            
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("\n提示：请确保你已经运行过量化脚本并生成了以下文件：")
        print(f"  - ./saved/{{model_name}}_reorder_index_wikitext2_{{mean/hessian}}.pt")
        print(f"  - ./saved/{{model_name}}_p6_num_wikitext2_{{mean/hessian}}.pt")
        print(f"  - ./saved/{{model_name}}_p8_num_wikitext2_{{mean/hessian}}.pt")
        print("\n你可以通过运行原脚本的量化部分来生成这些文件。")
        sys.exit(1)
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()