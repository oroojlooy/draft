import gradio as gr
import mdtex2html
import tempfile
import re
import os
from PIL import Image
import torch
import requests
from io import BytesIO

# os.environ['HTTP_PROXY'] = 'http://10.8.0.169:3128'
# os.environ['HTTPS_PROXY'] = 'http://10.8.0.169:3128'

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def get_llava_model(model_path='llava-hf/llava-1.5-13b-hf'):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name)

    return tokenizer, model, image_processor, context_len, model_name


def call_llava_model(tokenizer, model, image_processor, model_name, query: str='',
               conv_mode: str = 'llama_3', temperature: float=0.2, top_p=None,
               num_beams: int = 1, max_new_tokens:int = 512, images: list = []):
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in query:
        if model.config.mm_use_im_start_end:
            query = re.sub(IMAGE_PLACEHOLDER, image_token_se, query)
        else:
            query = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, query)
    else:
        if model.config.mm_use_im_start_end:
            query = image_token_se + "\n" + query
        else:
            if DEFAULT_IMAGE_TOKEN not in query:
                query = DEFAULT_IMAGE_TOKEN + "\n" + query

    if "llama-2" in model_name.lower():
        conv_mode_ = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode_ = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode_ = "mpt"
    else:
        conv_mode_ = "llava_v0"

    if conv_mode is not None and conv_mode_ != conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode_, conv_mode, conv_mode
            )
        )
    else:
        conv_mode = conv_mode_

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    images_tensor = process_images(images, image_processor, model.config).to(model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    print(images_tensor.shape)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[
                images_tensor,
            ],
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    return outputs


# model_path = 'llava-hf/llava-1.5-13b-hf'
# model_path = 'liuhaotian/llava-v1.5-13b'
model_path = "./checkpoints/llava-v1.5-7b"
tokenizer, model, image_processor, context_len, model_name = get_llava_model(model_path=model_path)

print(f'[!] init the 7b model over ...')

g_cuda = torch.Generator(device='cuda').manual_seed(13)

filter_value = -float('Inf')
min_word_tokens = 10
gen_scale_factor = 4.0
stops_id = [[835]]
ENCOUNTERS = 1
load_sd = True
generator = g_cuda
conv_mode = None
num_beams = 1

max_num_imgs = 1
max_num_vids = 1
height = 320
width = 576

max_num_auds = 1
max_length = 246


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_images_to_html(text, image_path_list = []):
    outputs = text
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines) + "<br>"
    if len(image_path_list) > 0:
        for i in image_path_list:
            text += f'<img src="./file={i}" style="display: inline-block;width: 250px;max-height: 400px;"><br>'
            outputs = f'<Image>{i}</Image> ' + outputs
    text = text[:-len("<br>")].rstrip() if text.endswith("<br>") else text
    return text, outputs



def save_image_to_local(image: Image.Image):
    # TODO: Update so the url path is used, to prevent repeat saving.
    if not os.path.exists('temp'):
        os.mkdir('temp')
    filename = os.path.join('temp', next(tempfile._get_candidate_names()) + '.jpg')
    image.save(filename)
    return filename


def re_predict(prompt_input, image_path, chatbot, top_p, temperature, history, modality_cache, max_new_token):
    # drop the latest query and answers and generate again

    q, a = history.pop()
    chatbot.pop()
    return predict(q, image_path, chatbot, top_p, temperature, history, modality_cache, max_new_token)


def predict(prompt_input, image_path, chatbot, top_p, temperature, history, modality_cache, max_new_token):
    # prepare the prompt
    if type(image_path) is str:
        image_path = [image_path]

    if image_path is not None:
        print('image_path: ', image_path)
        image = []
        for path in image_path:
            image += [Image.open(path).resize((224, 224)).convert('RGB')]
    else:
        image = []
        image_path = []

    history_image_paths = []
    prompt_history = ''
    if history is not None:
        if len(history) != 0:
            for idx, (q, a) in enumerate(history):
                prompt_history += f'### Human: {q}\n### Assistant: {a}\n###'
                if "<img src=" not in q:
                    continue
                text_splits = q.split("<img src=\"./file=")
                for text_split in text_splits:
                    if not ("display:" in text_split and ";max-height:" in text_split and "/tmp/gradio/" in text_split):
                        continue
                    history_image_path = text_split.split('\" style=')[0]
                    if history_image_path not in history_image_paths:
                        history_image_paths += [history_image_path]
                    if history_image_path not in image_path:
                        image_path += [history_image_path]
                        image += [Image.open(history_image_path).resize((224, 224)).convert('RGB')]
    if len(image_path) == 0:
        prompt_text = [{"role": "user", "content": prompt_history + prompt_input},]
    else:
        image_text = ""
        for id_ in range(0, len(image_path)):
            image_text += "<image>\n"
        prompt_text = prompt_history + image_text + prompt_input

    print('prompt_text: ', prompt_text)

    if len(image) > 0:
        response = call_llava_model(tokenizer, model, image_processor, model_name, prompt_text, conv_mode, temperature,
                                   top_p, num_beams, max_new_token, image)
        # response = call_phi3_model_dummy(model, processor, prompt_text, image, top_p, temperature, guidance_scale_for_img)
        print('text_outputs: ', response)
        image_path_to_show = list(set(image_path) - set(history_image_paths))
        user_chat, user_outputs = parse_images_to_html(prompt_input, image_path_to_show)
        chatbot.append(("<p>{}</p>".format(user_chat), "<p>{}</p>".format(response)))
        history.append(("<p>{}</p>".format(user_chat), ''.join(response).replace('\n###', '')))
    else:
        response = "There is no attached image."
        chatbot.append(("<p>{}</p>".format(prompt_input), "<p>{}</p>".format(response)))
        history.append(("<p>{}</p>".format(prompt_input), ''.join(response).replace('\n###', '')))
    return chatbot, history, modality_cache, None, None, None,


def reset_user_input():
    return gr.update(value='')


def reset_dialog():
    return [], []


def reset_state():
    return None, None, [], []


def upload_image(conversation, chat_history, image_input):
    input_image = Image.open(image_input.name).resize(
        (224, 224)).convert('RGB')
    input_image.save(image_input.name)  # Overwrite with smaller image.
    conversation += [(f'<img src="./file={image_input.name}" style="display: inline-block;">', "")]
    return conversation, chat_history + [input_image, ""]


with gr.Blocks() as demo:

    gr.HTML("""
        <h1 align="center" style=" display: flex; flex-direction: row; justify-content: center; font-size: 25pt; "><img src='./file=nextgpt.png' width="45" height="45" style="margin-right: 10px;">LLaVA</h1>
        <h3>This is the demo page of LLaVA, an image+text multimodal LLM</h3>
        <div style="display: flex;"><a href='https://huggingface.co/Efficient-Large-Model'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp  &nbsp  &nbsp <a href='https://github.com/haotian-liu/LLaVA'><img src='https://img.shields.io/badge/Github-Code-blue'></a> &nbsp &nbsp  &nbsp  <a href='https://arxiv.org/abs/2312.07533'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></div>
        """)

    with gr.Row():
        with gr.Column(scale=1, min_width=500):
            with gr.Row():
                chatbot = gr.Chatbot(label='LLaVA Chatbot', avatar_images=(
                (os.path.join(os.path.dirname(__file__), 'user.png')),
                (os.path.join(os.path.dirname(__file__), "bot.png"))))
                chatbot.height = 440
                # (os.path.join(os.path.dirname(__file__), "bot.png")))).style(height=440) # this does not work in v4.x

            with gr.Tab("User Input"):
                # with gr.Row(scale=3):
                with gr.Row():
                    user_input = gr.Textbox(label="Text", placeholder="Key in something here...", lines=3)
                # with gr.Row(scale=3):
                with gr.Row():
                    with gr.Column(scale=1):
                        # image_path = gr.UploadButton("🖼️ Upload Image", file_types=["image"])
                        image_path = gr.File(file_count="multiple", file_types=["image"])
                        # image_path = gr.Image(type="filepath", label="Image")  # .style(height=200)

        with gr.Column(scale=0.3, min_width=300):
            with gr.Group():
                with gr.Accordion('Text Advanced Options', open=True):
                    top_p = gr.Slider(0, 1, value=0.01, step=0.01, label="Top P", interactive=True)
                    temperature = gr.Slider(0, 1, value=0.0, step=0.01, label="Temperature", interactive=True)
                with gr.Accordion('Max new token', open=True):
                    max_new_token = gr.Slider(1, 1024, value=512, step=1, label="max new tokens",
                                                       interactive=True)
                    # num_inference_steps_for_img = gr.Slider(10, 50, value=50, step=1, label="Number of inference steps",
                    #                                         interactive=True)
            with gr.Tab("Operation"):
                # with gr.Row(scale=1):
                with gr.Row():
                    submitBtn = gr.Button(value="Submit & Run", variant="primary")
                # with gr.Row(scale=1):
                with gr.Row():
                    resubmitBtn = gr.Button("Rerun")
                # with gr.Row(scale=1):
                with gr.Row():
                    emptyBtn = gr.Button("Clear History")

    history = gr.State([])
    modality_cache = gr.State([])

    # predict(prompt_input, image_path, chatbot, top_p, temperature, history, modality_cache, max_new_token)
    submitBtn.click(
        predict, [user_input, image_path, chatbot, top_p, temperature, history, modality_cache, max_new_token,
            ], [chatbot, history, modality_cache, image_path,], show_progress=True)

    resubmitBtn.click(
        re_predict, [user_input, image_path, chatbot, top_p, temperature, history, modality_cache,
                     max_new_token,], [chatbot, history, modality_cache, image_path,]
                    , show_progress=True)

    submitBtn.click(reset_user_input, [], [user_input])
    emptyBtn.click(reset_state, outputs=[image_path, chatbot, history, modality_cache], show_progress=True)

demo.queue().launch(share=True, inbrowser=True, server_name='0.0.0.0', server_port=24011)

