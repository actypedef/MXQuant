# MXQuant

MXQuant-自然语言处理课程设计 


通过将 90% 的权重量化压缩为 MXFP4，10% 的关键权重量化为 MXFP6，仅需原本 30% 的显存占用


同时将激活值矩阵量化为 MXFP8 ，实现了预填充阶段接近 80% 的速度提升


在压缩与加速的条件下，模型在下游任务 0-shot 上维持 Llama3.1-8B 98.5% 的准确率


量化 CUDA Kernel 实现参考了 [MicroMix](https://github.com/lwy2020/MicroMix)


## 1. Installation
```bash
conda create -n mxquant python=3.10 -y
conda activate mxquant
```
Please make sure that [CUDA 12.8](https://developer.nvidia.com/cuda-12-8-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local) is in your environment.
```bash
git clone --recurse-submodules https://github.com/actypedef/MXQuant.git
cd MicroMix
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## 2. Usage

### 2.1 Preprocessing
Reorder_indices, p6_num, p8_num are needed for quantization:
```bash
python reorder_indices.py --model /PATH/TO/YOUR/MODEL/ --samples 32 --seqlen 2048 --act_sort_metric mean
```
Results are saved in saved/
### 2.2 Building Kernels
Please refer to `mgemm/README.md`
```bash
cd mgemm/
```
### 2.3 Zero-shot, Few-shot Accuracy and Perplexity Evaluation
```bash
bash test.sh /PATH/TO/YOUR/MODEL/
```

## 3. Efficiency Evaluation
MXQuant efficiency:
```bash
python benchmarks/benchmark_e2e_mxquant.py --model 'llama-3.1-8b' --batch_size 8 --prefill_seq_len 2048
```
FP16 efficiency:
```bash
python benchmarks/benchmark_e2e_fp16.py --model /PATH/TO/YOUR_MODEL --batch_size 8 --prefill_seq_len 2048
```
INT8 efficiency:
```bash
pip install bitsandbytes==0.47.0
python benchmarks/benchmark_e2e_int8.py --model /PATH/TO/YOUR_MODEL --batch_size 12 --prefill_seq_len 2048
```

## Acknowledagement

- [MicroMix](https://github.com/lwy2020/MicroMix)
- [Atom](https://github.com/efeslab/Atom.git)
- [QuaRot](https://github.com/spcl/QuaRot)
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer/tree/main)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
