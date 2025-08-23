


def prompt_one():

    base_prompt = "Role: You are a medical assistant aiding clinicians. \nInstruction: Identify atleast 3 (at max 5) most important clinical facts from Patient A's data that is crucial to deciding the Emergency Department (ED) Disposition of Patient A. Provided below is Patient A's data based on their clinical data and measurements collected in the ED. None indicates no findings. NA indicates not applicable or data not available.\nInput Data:\nPatient A - pat_detail. \nOutput Format: Structure your response where each line is one important clinical fact.\n"
    return base_prompt


# final_prompt = f"""Instruction: You are a medical assistant that needs to predict the Emergency Department (ED) Disposition of Patient A. To assist you in making this decision, you are given (i) Clinical Context from medical articles (ii) Past Patients clinical data and their ED Disposition  (iii) Patient A's Clinical Data.\n
# Clinical Context:\n
# {context}

# Here is the patient data for Patient A and for all the similar past patients. \n
# {prompt}

# Question:
# Based on the context above, please update the reasoning below if needed to make sure it is clinically relevant. Do not change the formatting:
# {final_answer}
# """


def prompt_return(type_learning, few_shot_number=None):
    """Generate the prompt for zero-shot or few-shot learning."""
    
    #Please provide a reasoning with your answer along with return one of two answers: '0: Patient should be discharged home' '1: Patient should be admitted to hospital'.


    if type_learning == 'zero':
        base_prompt="""Instruction: You are a medical assistant that needs to predict the Emergency Department (ED) Disposition of Patient A. To assist you in making this decision, you are given (i) Clinical Context from medical articles (ii) Patient A's Clinical Data in the ED.\n"""
    else:
        base_prompt="""Instruction: You are a medical assistant that needs to predict the Emergency Department (ED) Disposition of Patient A. To assist you in making this decision, you are given (i) Clinical Context from medical articles (ii) Past Patients clinical data and their ED Disposition  (iii) Patient A's Clinical Data in the ED.\n"""

    
    base_prompt+="Clinical Context:\n clinical_context"

    if type_learning=='zero':

        prompt=f"{base_prompt}\nInput Data:\nPatient A - pat_detail. \n\n"
        final_prompt=prompt+"Output Format: Structure your output in 3 sections. 1. Recap what are the at max 6 most important clinical facts from patient summary that relate to your decision. 2. Explain the actual reasoning behind the key facts and your decision about Patients A's ED Disposition. 3. Final Predicted ED Disposition for Patient A.\n"

    else:
        # Few-shot learning case
        few_shot_examples = [f"\nPrevious Patient {chr(66+n-1)} - pat{n}_detail" for n in range(1, few_shot_number + 1)]


        prompt=f"{base_prompt}\nInput Data: Here are a few examples of past patients data collected in the ED and their respective ED Disposition." + ".\n".join(few_shot_examples) + ".\n\nHere is Patient A's data:\n\nPatient A - pat_detail."
    


        final_prompt=prompt+"\n\nOutput Format: Structure your output in 4 sections. 1. Recap what are the at max 6 most important clinical facts from patient summary that relate to your decision. 2. If there are key comparisons with past patients presented, summarize them in this section. 3. Explain the actual reasoning behind the key facts and your decision about Patients A's ED Disposition. 4. Final Predicted ED Disposition for Patient A.\n"
            

    return final_prompt


'''
Structure your output in 4 sections. 1. Recap what are the at max 6 most important clinical facts from patient summary that relate to your decision. 2. If there are key comparisons with past patients presented, summarize them in this section. 3. Explain the actual reasoning behind the key facts and your decision about Patients A's ED Disposition. 4. Final Predicted ED Disposition for Patient A.

 Structure your output in 3 sections. 1. Recap what are the at max 6 most important clinical facts from patient summary that relate to your decision. 2. Explain the actual reasoning behind the key facts and your decision about Patients A's ED Disposition. 3. Final Predicted ED Disposition for Patient A.\n
Structure the output:

1. Recap what are the atleast 3 (at max 6) most important clinical facts that relate to our decision making precisely
2.  If there are key comparisons with previous patients, summarize them here. Dont repiculate the entire clinical presentation of the summary of past patients 
3. Now explain the actual reasoning behind the key facts and your decision about Patient A's ED Disposition.
4. Final ED Disposition Prediction: 

Keep your response concise and to a max of 250 words. 
'''





#\n\n If necessary, use the following past patient examples to guide your reasoning and prediction of Patient A's ED disposition. Explain how you using the past patients to guide your reasoning? 
#Use the following past patient examples to help guide your prediction of Patient A's ED disposition.