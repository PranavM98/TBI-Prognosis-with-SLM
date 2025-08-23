#nornmal Implementation
import faiss
import pickle
import numpy as np
# from sentence_transformers import SentenceTransformer
from prompts import prompt_return
import time
import os

import requests
import pandas as pd
import random
from tqdm import tqdm




'''
Create Command Line Arguments that 
1. select type (Zero Shot or Few Shot
2. If few shot - what number of k
3. Random or RAPID-TBI
'''


import argparse
random.seed(42)

# parser = argparse.ArgumentParser(description="ED Disposition - TBI Patients")

# # Add arguments
# parser.add_argument("--type", type=str, required=True, help="zero/few")
# parser.add_argument("--few_shot_number", type=int, help="1,3")
# parser.add_argument("--random", action="store_true", help='random or RAPID')
# parser.add_argument("--model", default='phi', type=str, help='phi/')


# # Parse arguments
# args = parser.parse_args()

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="ED Disposition - TBI Patients")
    parser.add_argument("--type", type=str, required=True, help="zero/few")
    parser.add_argument("--few_shot_number", type=int, help="1,3,5,7")
    parser.add_argument("--random", action="store_true", help='random or RAPID')
    parser.add_argument("--model", default='phi', type=str, help='phi/llama/qwen/gemma')
    parser.add_argument("--reason", type=str, help='no/normal/cot')
    parser.add_argument("--port", type=int, required=True, help="8000")
    parser.add_argument("--sample_size", type=int, default=1411, help="1411 (GCS ONLY) if everything")
    parser.add_argument("--trial", type=str, default='1', help="1")

    

    return parser.parse_args()


def save_results(dataframe, args):
    """Save the results to a CSV file based on the arguments."""
    file_suffix = f"random_k{args.few_shot_number}" if args.random else f"k{args.few_shot_number}"
    filename = f"{args.model}_{file_suffix}_{args.trial}.csv" if args.type == 'few' else f"{args.model}_k0_{args.trial}.csv"

    if args.reason=='no':
        folderpath='full_results/r0/'
    elif args.reason=='normal':
        folderpath = 'full_results/r1_normal/'
    elif args.reason=='cot':
        folderpath = 'full_results/r1_cot/'

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
    answers, response_times = [], []

    for i in tqdm(range(test_list)):
        # Generate the base prompt only once per iteration
        prompt = prompt_return(type_learning=args.type, few_shot_number=args.few_shot_number, reason=args.reason)

        # Replace placeholders with actual patient details
        pat = remove_label(patient_summaries[i])
        prompt = prompt.replace('pat_detail', pat)

        if args.type == 'few':
            neighbor_number = args.few_shot_number
            for n in range(1, neighbor_number + 1):
                neighbor_summary = (data[f'Neighbor{random.randint(1,7)}_Summary'].values[random.randint(0, test_list-1)]
                                    if args.random else data[f'Neighbor{n}_Summary'].values[i])
                prompt = prompt.replace(f'pat{n}_detail', neighbor_summary)

    
        # print(answer_text)
        # print("\n------------\n")
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
                final_answer= response.json()#.get("response", "")


            elif args.model=='gemma':
                start_time = time.time()
                response = requests.post(url, json={"question": prompt})
                elapsed_time = time.time() - start_time
                answer_text = response.json().get("response", "")
                final_answer=answer_text
            elif args.model=='phi_large':
                start_time = time.time()
                response = requests.post(url, json={"question": prompt})
                elapsed_time = time.time() - start_time  # Calculate response time
                # Extract answer
                final_answer = response.json()#.get("response", "")
                print(final_answer)
            else: #Phi and Phi Reasoning
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
        answers.append(final_answer)
        response_times.append(elapsed_time)

    return pd.DataFrame({
        'Patient Summaries': patient_summaries[:test_list],
        'Predicted Disposition': answers,
        'Response Time (s)': response_times
    })

# Main execution
if __name__ == "__main__":
    args = parse_arguments()
    print("Reason: ", args.reason)
    if args.model=='phi':
        print("Model: Phi")
        url = f"http://localhost:{args.port}/generate_phi"
    elif args.model=='phi_large':
        print("Model: Phi Large")
        url = f"http://localhost:{args.port}/generate_phi_large"

    elif args.model=='phi_reason':
        print("Model: Phi Reasoning")
        url = f"http://localhost:{args.port}/generate_phi_reasoning"
        
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

    if int(args.sample_size)==1411:
        # Process patient data and generate predictions
        predicted_df = process_patient_data(args, data,len(data))
    else:
        predicted_df = process_patient_data(args, data,int(args.sample_size))
    # Save results
    save_results(predicted_df, args)
    print("Saved Results")

