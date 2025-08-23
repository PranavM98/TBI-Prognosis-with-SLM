def prompt_return(reason, type_learning, few_shot_number=None):
    """Generate the prompt for zero-shot or few-shot learning."""
    
    #Please provide a reasoning with your answer along with return one of two answers: '0: Patient should be discharged home' '1: Patient should be admitted to hospital'.

    base_prompt = ("Role: You are a medical assistant aiding clinicians. \nInstruction: Predict the Emergency Department (ED) Disposition of Patient A based on their clinical details and measurements collected in the ED. Respond with either: 'Admit to Hospital' or 'Discharge Home'.\nContext: Patient data is collected in the ED and presented in text format. None indicates no findings. NA indicates not applicable or data not available.")


    if type_learning == 'zero':
        prompt=f"{base_prompt}\nInput Data:\nPatient A - pat_detail. \n\n"

    else:
        # Few-shot learning case
        few_shot_examples = [f"\nPrevious Patient {chr(66+n-1)} - pat{n}_detail" for n in range(1, few_shot_number + 1)]


        prompt=f"{base_prompt}\nInput Data: Here are a few examples of past patients data collected in the ED and their respective ED Disposition." + ".\n".join(few_shot_examples) + ".\n\nHere is Patient A's data:\n\nPatient A - pat_detail."
    

    if reason=='no':
        prompt=prompt+"\nOutput Format: Do not provide any reasoning. Conclude your response with the appropriate disposition decision: 'Discharge Home' or 'Admit to Hospital'.\n"
    
    elif reason=='normal':
        prompt=prompt+"\nOutput Format: Provide a concise explanation for Patient A's predicted ED disposition. Conclude your response with the appropriate disposition decision: 'Discharge Home' or 'Admit to Hospital'\n"


    elif reason=='cot':
        prompt=prompt+"\nOutput Format: Let think step-by-step for predicting Patient A's Predicted ED Disposition. Conclude your response with the appropriate disposition decision: 'Discharge Home' or 'Admit to Hospital'\n"
    

    return prompt


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