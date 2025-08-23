
## RAG IMplementation

import faiss
import pickle
import numpy as np
# from sentence_transformers import SentenceTransformer
from partA_prompts_RAG import prompt_return, prompt_one
import time
import os
import re
import requests
import pandas as pd
import random
from tqdm import tqdm

import doc_rag_multiple_new

import numpy as np
from scipy import stats

'''
Create Command Line Arguments that 
1. select type (Zero Shot or Few Shot
2. If few shot - what number of k
3. Random or RAPID-TBI
'''

# List all your chunk files
chunk_files = [
    '../Doc_RAG/nswh_output/chunks/chunks.json',
    '../Doc_RAG/acs_output/chunks/chunks.json',
    '../Doc_RAG/btf_output/chunks/chunks.json',
    '../Doc_RAG/cma_output/chunks/chunks.json',
    '../Doc_RAG/nhice_output/chunks/chunks.json',
    '../Doc_RAG/scn_output/chunks/chunks.json'
]

# Create multi-database RAG system
rag_system = doc_rag_multiple_new.MultiDatabaseSentenceRAGSystem(chunks_files=chunk_files)





import argparse
random.seed(42)



def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ED Disposition - TBI Patients")
    parser.add_argument("--type", type=str, required=True, help="zero/few")
    parser.add_argument("--few_shot_number", type=int, help="1,3,5,7")
    parser.add_argument("--random", action="store_true", help='random or RAPID')
    parser.add_argument("--model", default='phi', type=str, help='phi/llama/qwen/gemma')
    parser.add_argument("--port", type=int, required=True, help="8000")
    parser.add_argument("--sample_size", type=int, default=30, help="2330 if everything")
    parser.add_argument("--max_chunk", type=int, default=10, help="Global Max chunks")
    parser.add_argument("--top_k", type=int, default=1, help="Sentence top K chunks")
    parser.add_argument("--trial", type=int, default=1, help="Trial")
    parser.add_argument("--rag_strategy", type=str, default='sentence-local', help="sentence-local, sentence-global, prompt-global")
    
    
    
    

    

    return parser.parse_args()


def save_results(dataframe, args):
    """Save the results to a CSV file based on the arguments."""
    #add trial number

    file_suffix = f"random_k{args.few_shot_number}" if args.random else f"k{args.few_shot_number}"
    filename = f"{args.model}_{file_suffix}_st{args.top_k}_trial{args.trial}.csv" if args.type == 'few' else f"{args.model}_k0_st{args.top_k}_trial{args.trial}.csv"


    folderpath=f'full_results_RAGpartA/sentence_topk/top_k{args.top_k}/'

    # Create the folder if it doesn't exist
    os.makedirs(folderpath, exist_ok=True)

    finalpath = os.path.join(folderpath, filename)
    dataframe.to_csv(finalpath, index=False)

def remove_label(text):
    """Remove labels from the patient summary."""
    return text[:text.find('ED Disposition')-2]


def process_patient_data(args, data, test_list=5):
    """Process patient data, generate prompts, and collect responses."""
    patient_summaries = data['Test_Summary'].values
    print("Test List: ", test_list)
    answers, response_times, prompts = [], [], []
    final_RAG_response_times=[]
    for i in tqdm(range(test_list)):
        # Generate the base prompt only once per iteration
        prompt = prompt_one()

        # Replace placeholders with actual patient details
        pat = remove_label(patient_summaries[i])        
        prompt = prompt.replace('pat_detail', pat)

        final_RAG_start=time.time()

        try:
            if args.model=='qwen':
                # Measure API response time
                start_time = time.time()
                response = requests.post(url, json={"question": prompt})
                elapsed_time = time.time() - start_time  # Calculate response time
                # Extract answer
                final_answer= response.json().get("response", "")
            
            elif args.model=='llama':
                # Measure API response time
                start_time = time.time()
                response = requests.post(url, json={"question": prompt})
                elapsed_time = time.time() - start_time  # Calculate response time
                # Extract answer
                final_answer= response.json().get("response", "")['content']
            elif args.model=='gemma':
                start_time = time.time()
                response = requests.post(url, json={"question": prompt})
                elapsed_time = time.time() - start_time
                answer_text = response.json().get("response", "")
                final_answer=answer_text

            else:
                # Measure API response time
                start_time = time.time()
                response = requests.post(url, json={"question": prompt})
                
                elapsed_time = time.time() - start_time  # Calculate response time
                # Extract answer
                final_answer = response.json().get("response", "")


        except:
            final_answer='N/A'
            elasped_time='N/A'
        # Append results

        if args.rag_strategy=='sentence-global'
            results =rag_system.query(prompt=final_answer,
                    top_k_per_sentence=args.top_k)

            # Get formatted context for LLM generation
            context = rag_system.get_context_for_generation(results, max_chunks=args.max_chunk)
        
        elif args.rag_strategy=='sentence-local':
            #NEW: Get top-k per sentence without global deduplication
            results = doc_rag_multiple_new.search_per_sentence_topk(rag_system, final_answer, top_k=args.top_k)

            # Get context for generation (organized by sentence)
            context = rag_system.get_context_for_generation_per_sentence(results)
        elif args.rag_strategy=='prompt-global':

            # NEW: Search using whole prompt (no sentence breakdown)
            results = doc_rag_multiple_new.search_whole_prompt(rag_system, final_answer, top_k=args.top_k)

            # Get context for generation from whole prompt search
            context = rag_system.get_context_for_generation(results)


        prompt_final=prompt_return(type_learning=args.type, few_shot_number=args.few_shot_number)


        
        pat = remove_label(patient_summaries[i])        
        prompt_final = prompt_final.replace('pat_detail', pat)
        prompt_final = prompt_final.replace('clinical_context', context)
        
        if args.type == 'few':
            neighbor_number = args.few_shot_number
            for n in range(1, neighbor_number + 1):
                neighbor_summary = (data[f'Neighbor{random.randint(1,7)}_Summary'].values[random.randint(0, test_list-1)]
                                    if args.random else data[f'Neighbor{n}_Summary'].values[i])
                prompt_final = prompt_final.replace(f'pat{n}_detail', neighbor_summary)


        response = requests.post(url, json={"question": prompt_final})
        
        
        # Extract answer
        updated_answer = response.json().get("response", "")

        answers.append(updated_answer)
        prompts.append(prompt_final)


    return pd.DataFrame({
        'Patient Summaries': prompts,
        'Predicted Disposition': answers
    })

# Main execution
if __name__ == "__main__":
    args = parse_arguments()
    if args.model=='phi':
        print("Model: Phi")
        url = f"http://localhost:{args.port}/generate_phi"
        
    elif args.model=='qwen':
        print("Model: Qwen")
        url = f"http://localhost:{args.port}/generate_qwen"

    elif args.model=='gemma':
        print("Model: Gemma")
        url = f"http://localhost:{args.port}/generate_gemma"
    
    elif args.model=='llama':
        print("Model: Llama")
        url = f"http://localhost:{args.port}/generate_llama"

    
    print(f"{args.type.capitalize()} Shot Learning")
    print(f"Random: {args.random}" if args.type == 'few' else "")

    # Load data
    data = pd.read_csv("/cwork/pm231/data_neighbors_with_reports_gcs_only.csv")


    predicted_df = process_patient_data(args, data,len(data))
    #Save results
    save_results(predicted_df, args)
    print("Saved Results")

