from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

import torch
from transformers import BitsAndBytesConfig, Gemma3ForCausalLM, Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
import argparse
torch.random.manual_seed(0)
torch.cuda.empty_cache()
parser = argparse.ArgumentParser(description="LLM Server")

# Add arguments
parser.add_argument("--model", type=str, required=True, help="phi/gemma/llama/qwen")
parser.add_argument("--port", type=int, required=True, help="8000")


# Parse arguments
args = parser.parse_args()
access_token = '**************************'


if args.model=='phi':

    model_path = "microsoft/Phi-4-mini-instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True)#,token=access_token

    tokenizer = AutoTokenizer.from_pretrained(model_path)#,token=access_token)

elif args.model=='phi_reason':

    model_id = "microsoft/Phi-4-mini-reasoning"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,token=access_token
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id,token=access_token)

elif args.model=='medllama3':

  
    MODEL_NAME = "johnsnowlabs/JSL-MedLlama-3-8B-v2.0"
    print("Loading MedLlama-3 model...")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Create text generation pipeline
    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",max_new_tokens=1024
    )


elif args.model=='gemma':
    model_id = "google/gemma-3-1b-it"

    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    # quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = Gemma3ForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto", token=access_token).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)

elif args.model=='qwen':
    model_name = "Qwen/Qwen2.5-3B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

elif args.model=='llama':
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    )

    # # Load the tokenizer and model
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct",token=access_token)
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct",torch_dtype="auto",device_map="auto",token=access_token)


# Define request format
class PromptRequest(BaseModel):
    question: str

# # Load the model and tokenizer
# Initialize FastAPI app
app = FastAPI()
# Create text generation pipeline



@app.post("/generate_phi_reasoning")
def generate_quen(request: PromptRequest):
    messages = [{
        "role": "user",
        "content": request.question
    }]   
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    outputs = model.generate(
        **inputs.to(model.device),
        max_new_tokens=32768,
        temperature=0.01,
        top_p=0.95,
        do_sample=True,
    )
    outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])

    print(outputs[0])




@app.post("/generate_qwen")
def generate_quen(request: PromptRequest):
    messages = [
        {"role": "user", "content": request.question}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
    return {"response": response}




@app.post("/generate_gemma")
def generate_text_gemma(request: PromptRequest):
    # Format the prompt according to Gemma 3's expected input

    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a medical assistant."},]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": request.question}]
            },
        ],
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device).to(torch.bfloat16)


    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=1024)

    outputs = tokenizer.batch_decode(outputs)
    match = re.search(r"<start_of_turn>model\n(.*?)<end_of_turn>", outputs[0], re.DOTALL)
    if match:

        model_response = match.group(1).strip()

        return {"response": model_response}

    # model_inputs = tokenizer(request.question, return_tensors="pt").to(model.device)

    # input_len = model_inputs["input_ids"].shape[-1]

    # with torch.inference_mode():
    #     generation = model.generate(**model_inputs, max_new_tokens=1028, do_sample=False)
    #     generation = generation[0]#[input_len:]

    # decoded = tokenizer.decode(generation, skip_special_tokens=True)
    # return {"response": decoded}

@app.post("/generate_llama")
def generate_text_llama(request: PromptRequest):

    # # Generate text
    # input_text = request.question
    # inputs = tokenizer(input_text, return_tensors="pt")
    # outputs = model.generate(inputs["input_ids"], max_length=1028)

    # # Print the result
    # print(tokenizer.decode(outputs[0]))

    messages = [
        {"role": "system", "content": "You are a medial assistant"},
        {"role": "user", "content": request.question},
    ]
    outputs = pipe(
        messages,
        max_new_tokens=1024,
    )
    # print(outputs[0]["generated_text"][-1])
    return {"response":outputs[0]["generated_text"][-1]}



@app.post("/generate_phi")
def generate_text_phi(request: PromptRequest):

    messages = [
    {"role": "user", "content": request.question}]
  
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    
    generation_args = {
        "max_new_tokens": 1024,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }
    
    output = pipe(messages, **generation_args)
    return {"response":output[0]['generated_text']}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
