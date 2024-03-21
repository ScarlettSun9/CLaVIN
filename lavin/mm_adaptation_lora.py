import torch
import json
from lavin import ModelArgs, Tokenizer, Transformer
from lavin import Transformer_LoRA
from lavin.mm_adapter import set_MMAdapter,set_Clip_Adapter
from pathlib import Path
from util.apply_delta import apply_model_delta_online
from peft import LoraConfig, get_peft_model, cast_mixed_precision_params

def _load_and_redistribute_checkpoint(llama_model_path, tokenizer_path, model_name):

    with open(Path(llama_model_path) / 'params.json') as f:
        params = json.load(f)
    tokenizer = Tokenizer(model_path=str(Path(tokenizer_path) / 'tokenizer.model'))
    print('Using model path: %s, model_name: %s' % (llama_model_path, model_name))
    if model_name=='7B':
        checkpoint = torch.load(llama_model_path + '/consolidated.00.pth', map_location="cpu")
        return checkpoint, tokenizer, params


    checkpoints = (Path(llama_model_path)).glob('*.pth')
    checkpoints = sorted(checkpoints)


    loaded = []
    for x in checkpoints:
        print('loading from', x)
        loaded.append(torch.load(x, map_location='cpu'))

    full_state_dict = {}
    split_dims = {}

    def add_weight_with_split_dim(name, dim):
        if dim < 0:  # bcast without split
            full_state_dict[name] = loaded[0][name].clone()
        else:
            full_state_dict[name] = torch.cat([x[name] for x in loaded], dim=dim)
        for x in loaded:
            del x[name]
        split_dims[name] = dim

    add_weight_with_split_dim('tok_embeddings.weight', 1)
    add_weight_with_split_dim('norm.weight', -1)
    add_weight_with_split_dim('output.weight', 0)
    for i in range(params['n_layers']):
        print('gathering layer %d of %d' % (i, params['n_layers']))
        layer_prefix = f'layers.{i}.'
        bcast_names = [
            'attention_norm.weight',
            'ffn_norm.weight',
        ]
        column_parallel_names = [
            'attention.wq.weight',
            'attention.wk.weight',
            'attention.wv.weight',
            'feed_forward.w1.weight',
            'feed_forward.w3.weight',
        ]
        row_parallel_names = [
            'attention.wo.weight',
            'feed_forward.w2.weight',
        ]
        for key in bcast_names:
            add_weight_with_split_dim(layer_prefix + key, -1)
        for key in column_parallel_names:
            add_weight_with_split_dim(layer_prefix + key, 0)
        for key in row_parallel_names:
            add_weight_with_split_dim(layer_prefix + key, 1)

    checkpoint=full_state_dict


    return checkpoint, tokenizer, params

def LaVIN_LoRA(args):
    
    llama_model_path = args.llama_model_path
    model_name = args.llm_model
    tokenizer_path = args.tokenizer_path
    
    checkpoint, tokenizer, params = _load_and_redistribute_checkpoint(llama_model_path, tokenizer_path, model_name)
    
    model_args: ModelArgs = ModelArgs(
        max_seq_len = args.max_seq_len, max_batch_size=32,
        hidden_proj = args.hidden_proj, drop_path = args.drop_path, **params
    )
    
    model_args.vocab_size = tokenizer.n_words
    
    if args.cpu_load:
        torch.set_default_tensor_type(torch.HalfTensor)
    else:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
    
    llama = Transformer_LoRA(model_args)
    
    # delete language encoder
    del llama.backbone.transformer
    
    """
    print(llama)
    for name, param in llama.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Trainable: {param.requires_grad}")
    """
        
    print('\n\n\n')

    
    # Config for the LoRA Injection via PEFT
    lora_config = LoraConfig(r=2, # rank dimension of the LoRA injected matrices
                             lora_alpha=8, # parameter for scaling
                             target_modules=['wq', 'wv'], # be precise about dense because classifier has dense too
                             lora_dropout=0.1, # dropout probability for layers
                             bias="all", # none, all, or lora_only
                            )
    
    llama = get_peft_model(llama, lora_config)
    
    for param in llama.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    
    # cast_mixed_precision_params(llama, dtype=torch.float16)
            
    llama.print_trainable_parameters()
    print('\n\n\n')
    
    
    torch.set_default_tensor_type(torch.FloatTensor)
    
    if args.bits in ['4bits', '8bits']:
        from util.quantization import quant_model_bnb
        llama.layer = quant_model_bnb(llama.layers, quant_bit = args.bits)
    
    llama.load_state_dict(checkpoint, strict=False)
    
    return llama