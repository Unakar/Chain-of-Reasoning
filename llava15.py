import os
import sys
sys.path.append("./models")
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
from PIL import Image
import requests
import torch
import torch.nn.functional as F
from io import BytesIO
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

def load_image(image_path_or_url):
    if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path_or_url).convert('RGB')
    return image

# ===> specify the model path
model_path = "liuhaotian/llava-v1.5-7b"

# load the model
load_8bit = False
load_4bit = False
device = "cuda:0" if torch.cuda.is_available() else "cpu"

disable_torch_init()

model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, 
    None, # model_base
    model_name, 
    load_8bit, 
    load_4bit, 
    device=device
)

# ===> specify the image path or url and the prompt text
image_path_or_url = "/home/t2vg-a100-G4-43/VLM-Visualizer/image.png" #"https://github.com/open-compass/MMBench/blob/main/samples/MMBench/1.jpg?raw=true"
prompt_text = "QUESTION. How many countries are there in the photo? List their names " #Options:A. east B. west C. north D. south

################################################
# preparation for the generation
if "llama-2" in model_name.lower():
    conv_mode = "llava_llama_2"
elif "mistral" in model_name.lower():
    conv_mode = "mistral_instruct"
elif "v1.6-34b" in model_name.lower():
    conv_mode = "chatml_direct"
elif "v1" in model_name.lower():
    conv_mode = "llava_v1"
elif "mpt" in model_name.lower():
    conv_mode = "mpt"
else:
    conv_mode = "llava_v0"

conv = conv_templates[conv_mode].copy()
if "mpt" in model_name.lower():
    roles = ('user', 'assistant')
else:
    roles = conv.roles

image = load_image(image_path_or_url)
image_tensor, images = process_images([image], image_processor, model.config)
image = images[0]
image_size = image.size
if type(image_tensor) is list:
    image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
else:
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

if model.config.mm_use_im_start_end:
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt_text
else:
    inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt_text

conv.append_message(conv.roles[0], inp)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
# remove system prompt for better clarity
prompt = prompt.replace(
    "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. ",
    ""
)

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
display(image)
print(prompt_text)

# 第一次推理，保存第15层的image token hidden states
with torch.inference_mode():
    outputs_first = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image_size],
        do_sample=False,
        max_new_tokens=512,
        use_cache=True,
        return_dict_in_generate=True,
        output_attentions=True,
        output_hidden_states=True,  # 关键：输出hidden states
    )

text_first = tokenizer.decode(outputs_first["sequences"][0]).strip()
print("First run output:", text_first)

# 获取image token位置

# 假设image token使用DEFAULT_IMAGE_TOKEN和DEFAULT_IM_START_TOKEN,DEFAULT_IM_END_TOKEN
# 一般来说，image tokens可能是单个special token，根据您的tokenizer而定
# 这里假设DEFAULT_IMAGE_TOKEN在input_ids中对应的id是image token的位置
image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
image_token_positions = (input_ids[0] == image_token_id).nonzero().squeeze(-1)

# hidden_states是tuple，每一项对应一层的输出，形状为[batch, seq_len, hidden_size]
# layer的计数一般为0-based，所以第15层hidden state应该是 hidden_states[15]
# 请根据实际层数范围确认这里的索引，一般LLaMA有32层(0-31)
target_layer = 0
print(image_token_positions)
image_states = outputs_first["hidden_states"][target_layer][0][:, image_token_positions, :].clone()



def replace_image_tokens_hook(module, input, output):
    # output 是一个tuple，比如 (hidden_states, past_key_values)
    hidden_states = output[0]  # 取出hidden_states部分
    
    # 确保hidden_states是Tensor
    hidden_states = hidden_states.clone()
    hidden_states[:, image_token_positions, :] = image_states
    
    # 将处理后的hidden_states和其他输出再次组成tuple返回
    # 如果原本output有多个元素，如 (hidden_states, past_key_values, ...)，请保持对应长度
    # 比如：return (hidden_states, ) + output[1:]
    
    if len(output) == 2:
        # 典型情况： (hidden_states, past_key_values)
        return (hidden_states, output[1])
    else:
        # 如果返回结构更复杂，需要根据实际情况处理
        return (hidden_states,) + output[1:]


# 注册hook
hook_handles = []
for layer_idx in range(3, 26):
    # 假设model.model.layers是transformer block list
    handle = model.model.layers[layer_idx].register_forward_hook(replace_image_tokens_hook)
    hook_handles.append(handle)

# 第二次推理
with torch.inference_mode():
    outputs_second = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image_size],
        do_sample=False,
        max_new_tokens=512,
        use_cache=True,
        return_dict_in_generate=True,
        output_attentions=True,
        # 是否需要hidden_states看情况
        # output_hidden_states=True
    )

text_second = tokenizer.decode(outputs_second["sequences"][0]).strip()
print("Second run output:", text_second)


