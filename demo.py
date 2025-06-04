# -*- coding: utf-8 -*-
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

def compute_full_extracted_triples(intput_sentence):
    #v1.7
    def generate_prompt(text):
        promptmessage = [
        {
        "role": "user",
        "content": 
        '''Extract cyber attack related triples from text. Each triple must have exactly 3 elements: [Subject, Relation, Object]. Output format: {source sentence:[[subject1, relation1, object1],...]}. Use ellipsis in source sentence. Example: "The malware ... the system".'''
        },
        {
        "role": "assistant",
        "content": "I understand."
        },
        {
        "role": "user",
        "content": "Example: \"Leafminer attempts to infiltrate target networks through various means of intrusion: watering hole websites, vulnerability scans of network services on the internet, and brute-force/dictionary login attempts.\""
        },
        {
        "role": "assistant",
        "content": "{Leafminer attempts ... of intrusion: watering hole websites, vulnerability scans of network services on the internet, and brute-force/dictionary login attempts:[[SUBJECT:Leafminer,RELATION:attempts to infiltrate,OBJECT:target networks],[SUBJECT:Leafminer,RELATION:use,OBJECT:watering hole websites],[SUBJECT:Leafminer,RELATION:use,OBJECT:vulnerability scans of network services on the internet],[SUBJECT:Leafminer,RELATION:use,OBJECT:brute-force],[SUBJECT:Leafminer,RELATION:use,OBJECT:dictionary login attempts]]}."
        },
        {
        "role": "user",
        "content": "Example: \"Kismet is also a powerful tool for penetration testers that need to better understand their target and perform wireless LAN discovery.\""
        },
        {
        "role": "assistant",
        "content": "{Kismet is ... wireless LAN discovery.:[[SUBJECT:Kismet,RELATION:is a powerful tool for, OBJECT:penetration testers],[SUBJECT:testers, RELATION:understand, OBJECT:their target],[SUBJECT:testers,RELATION: perform, OBJECT:wireless LAN discovery]]}."
        },
        {"role": "user",
        "content": "Extract triples from: "+text}
        ]
        return promptmessage

    def generate_prompt_basedon3(inSent,inlist):
        promptmessage = [
        {
        "role": "user",
        "content":'Combine three entity extraction results into one. Each triple must have 3 elements: [SUBJECT:subject, RELATION:relation, OBJECT:object]. Output format: {source sentence:[[subject1, relation1, object1],...]}. Combine identical sentences with ellipsis. Input sentence: '+str(inSent)+', extracted triples: '+str(inlist)
        },
        ]
        return promptmessage
    
    def generate_prompt_postprocess(text):
        promptmessage = [
        {
        "role": "user",
        "content": 
        '''Modify triples following these rules:
1. Replace pronouns with specific names
2. Remove suffixes from malware/Trojan/CVE/hacking org subjects
3. Split complex triples into simpler ones
4. Create new triples from relation objects
5. Simplify objects to concise expressions
6. Remove modifiers and adjectives
7. Convert plural/past tense to singular/present
8. Remove prefixes from identifiers
9. Remove suffixes from proper nouns
10. Use SUBJECT:, RELATION:, OBJECT: prefixes

Input: '''+str(text)
        },
        ]
        return promptmessage

    def get_only_triples(text):
        text = text.replace(': [', ':[') 
        if "{" in text and "}" in text:
            start_index = text.rindex('{')
            end_index = text.rindex('}') + 1
            triple_only_text = text[start_index:end_index]
            if ":[" in triple_only_text and ']' in triple_only_text:
                source_sentence = triple_only_text.split(':')[0] 
                source_sentence=source_sentence.split('{')[1]
                words = source_sentence.split() 
                if len(words) <= 2:
                    abbreviation = source_sentence
                else:
                    abbreviation = " ".join(words[:2]) + " ... " + " ".join(words[-2:])
                triple_only_text = triple_only_text.replace(source_sentence, abbreviation)     
        else:
            text = text.replace('\n', '')
            
            if ":[" in text and "]" in text:
                start_index = text.rindex(':[')
                end_index = text.rindex(']') + 1
                triple_only_text = text[start_index:end_index]
            else:
                if "[["in text and "]]" in text:
                    start_index = text.rindex('[[')
                    end_index = text.rindex(']]') + 1
                    triple_only_text = text[start_index:end_index]
                else:
                    triple_only_text = text
        return triple_only_text
    
    def clean_text(text):
        import string,re
        if not isinstance(text, str):
            return text
        cleaned_text = re.sub(r'[^\x20-\x7E]', '', text)
        cleaned_text = re.sub(r'[\s{}]+'.format(re.escape(string.punctuation)), '', cleaned_text)
        cleaned_text = re.sub(r'SUBJECT|RELATION|OBJECT', '', cleaned_text)
        return cleaned_text if cleaned_text else 'Null'
    
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1"
    )
    model = "01-ai/Yi-1.5-6B-Chat"

    import concurrent.futures

    import re
    from tqdm import tqdm

    single_sentence=intput_sentence
    import ast
    content_first_extraction=''
    tried_times=0
    redoneflag=True
    first_answer_list=[]
    temperature_list=[1,0.5,0.2]
    
    while redoneflag:
        completion = client.chat.completions.create(
            model=model,
            messages=generate_prompt(single_sentence),
            max_tokens=1024,
            temperature=temperature_list[tried_times],
        )
        content_first_extraction = completion.choices[0].message.content
        cleaned_text = clean_text(str(content_first_extraction))
        
        if any(keyword in cleaned_text for keyword in ['CVExxx', 'Formbook', 'XLoader', 'Malwaresavetextfile', 'Leafminer', 'FinSpy', 'Kismet', 'Specificnamesofa']):
            first_answer_list.append('ERROR')
        else:
            first_answer_list.append(get_only_triples(content_first_extraction))
        tried_times =1+tried_times
        if tried_times > 2:
            redoneflag = False
        else:
            redoneflag = True

    completion = client.chat.completions.create(
        model=model,
        messages=generate_prompt_basedon3(single_sentence,first_answer_list[0:3]),
        max_tokens=1024,
        temperature=0.7,
    )
    content_first_extraction_merged=completion.choices[0].message.content
    content_first_extraction_merged=get_only_triples(content_first_extraction_merged)
 
    completion = client.chat.completions.create(
        model=model,
        messages=generate_prompt_postprocess(content_first_extraction_merged),
        max_tokens=1024,
        temperature=0.7,
    )
    content_simple_version = completion.choices[0].message.content
    extracted_text = get_only_triples(content_simple_version)
    print("\nEXTRACTED TEXT:", extracted_text)
    return extracted_text

def clean_full_extracted_triples(text):
    #remvoe all \n
    text=text.replace('\n','')
    #remove the space between ':" and "["
    import re
    text=re.sub(r':\s+\[',r'[',text)
    #remove the space between '[" and "["
    text=re.sub(r'\s+\[',r'[',text)
    #remove the space between ']" and "]"
    text=re.sub(r'\s+\]',r']',text)
    #replace ], ] or ],] or ] ,] with ]]
    text = re.sub(r'\]\s*,\s*\]', ']]', text)

    triple_only_text=text
    #if [[ and ]] in text, extract the content between them with a [ and ]
    if "[[" in text and "]]" in text:
        start_index = text.rindex('[[')+1
        end_index = text.rindex(']]') + 1
        triple_only_text = text[start_index:end_index]
    else:
        if "[["in text or "]]" in text:
            start_index = text.index('[')
            end_index = text.rindex(']') + 1
            triple_only_text = text[start_index:end_index]
    #remove all " and ' in text
    triple_only_text=triple_only_text.replace('"','')
    triple_only_text=triple_only_text.replace("'",'')
    #remove all the and The and THE in text
    #triple_only_text=triple_only_text.replace('the','')
    #triple_only_text=triple_only_text.replace('The','')
    #triple_only_text=triple_only_text.replace('THE','')
    return triple_only_text

def merge_extracted_triples(longmem,shortmem,sentence):
    def generate_prompt(longmem,shortmem,sentence):
        promptmessage = [
        {
        "role": "user",
        "content": 
        '''Merge triples from long-term and short-term memory. Rules:
1. Unify identical terms by removing prefixes/suffixes/modifiers
2. Use specific names for malware/CVE/Trojans/hacker orgs
3. Don't create new triples
4. Don't add example words
5. Output format: {source sentence:[[subject1, relation1, object1],...]}

Long-term memory: '''+str(longmem)+'''
Short-term memory: '''+str(shortmem)
        },
        ]
        return promptmessage

    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1"
    )
    model = "01-ai/Yi-1.5-6B-Chat"

    import concurrent.futures
    def clean_text(text):
        import string,re
        if not isinstance(text, str):
            return text
        cleaned_text = re.sub(r'[^\x20-\x7E]', '', text)
        cleaned_text = re.sub(r'[\s{}]+'.format(re.escape(string.punctuation)), '', cleaned_text)
        cleaned_text = re.sub(r'SUBJECT|RELATION|OBJECT', '', cleaned_text)
        return cleaned_text if cleaned_text else 'Null'
    import re
    from tqdm import tqdm
    retry_times=0
    pass_flag=False
   
    completion = client.chat.completions.create(
        model=model,
        messages=generate_prompt(longmem,shortmem,sentence),
        max_tokens=1024,
        temperature=0.7,)
    
    fullanswer = completion.choices[0].message.content   
         
    return fullanswer   

def check_brackets(my_string):
    if my_string is None or len(my_string) == 0:
        return False
    my_string = my_string.strip()
    first_char_is_bracket = my_string[0] == '['
    last_char_is_bracket = my_string[-1] == ']'

    if first_char_is_bracket and last_char_is_bracket:
        return True
    else:
        return False


def checker(my_string):
    promptmessage = [{
        "role": "user",
        "content":'You are a result checker. You are responsible for checking the result from other AI assistants. The AI assistant may say that " I am sorry, but I am Chat AI model and I am not able to do the task " or " You should do it by yourself" or "I am sorry, but I am not able to do the task". If you found those words or words with simlar meaning, you must reply me "ERROR", other wise, you should reply me "OK". Here is the result from other AI assistant: '+str(my_string)}]
    
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1"
    )
    model = "01-ai/Yi-1.5-6B-Chat"

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=promptmessage,
            max_tokens=128,
            temperature=1,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error in checker: {e}")
        return "ERROR"

def full_text_to_parts(text):
    import nltk
    # Split paragraphs
    paragraphs = text.split('\n')
    
    # Initialize result list
    processed_paragraphs = []

    # Process each paragraph
    for paragraph in paragraphs:
        if len(paragraph) > 600:
            # Use nltk to split sentences
            sentences = nltk.sent_tokenize(paragraph)
            # Truncate each sentence to 500 characters
            sentences = [x[0:500] for x in sentences]
            # Initialize new paragraph
            new_paragraph = ''
            for sentence in sentences:
                # Predicted length after merging
                temp_length = len(new_paragraph) + len(sentence)
                if temp_length < 600:
                    # If the length after merging is less than 600, add to the new paragraph
                    new_paragraph += (sentence + '\n')
                else:
                    # If the new paragraph has at least 20 characters, add it to the result list
                    if len(new_paragraph) >= 20:
                        processed_paragraphs.append(new_paragraph.strip())
                    # Reset new paragraph
                    new_paragraph = sentence + '\n'

            # Add the last new paragraph (if there is one and it has at least 20 characters)
            if len(new_paragraph) >= 20:
                processed_paragraphs.append(new_paragraph.strip())
        else:
            if len(paragraph) >= 20:
                processed_paragraphs.append(paragraph.strip())

    # Merge shorter paragraphs
    combined_paragraphs = []
    current_combined = ''

    for paragraph in processed_paragraphs:
        # Calculate the potential length of the current merged paragraph
        temp_length = len(current_combined) + len(paragraph) + 1  # Add 1 because there is a space between paragraphs
        if temp_length < 600:
            # If the length after merging is less than 600, continue to merge
            current_combined += (' ' if current_combined else '') + paragraph
        else:
            # Otherwise, store the current merged paragraph and reset
            combined_paragraphs.append(current_combined)
            current_combined = paragraph
            
    # Add the last merged paragraph (if there is one)
    if current_combined:
        combined_paragraphs.append(current_combined)

    return combined_paragraphs

def full_article_to_longmem(single_article):
    grouped_texts_strings = full_text_to_parts(single_article)
    triple_cache = []
    text_cache = []
    for i in range(len(grouped_texts_strings)):
        this_time_test=grouped_texts_strings[i]
        if len(this_time_test) > 1500:
            this_time_test = this_time_test[0:1500]
        print('Thinking about paragraph '+str(i))
        print('Seeing text：',this_time_test)
        triple = compute_full_extracted_triples(this_time_test)
        clean_triple_forMEM = clean_full_extracted_triples(triple)

        if  'Formbook' in clean_triple_forMEM or 'XLoader' in clean_triple_forMEM or 'savetextfile' in clean_triple_forMEM or 'Leafminer' in clean_triple_forMEM or 'FinSpy' in clean_triple_forMEM or 'Kismet' in clean_triple_forMEM or 'Agumon' in clean_triple_forMEM or 'Gabumon' in clean_triple_forMEM or 'Biyomon' in clean_triple_forMEM or '2042' in clean_triple_forMEM or check_brackets(clean_triple_forMEM)==False or checker(triple)=='ERROR':
            print('Current short-term memory does not meet requirements',triple)
            print('Retry extracting text 1')
            triple = compute_full_extracted_triples(this_time_test)
            clean_triple_forMEM = clean_full_extracted_triples(triple)

        if  'Formbook' in clean_triple_forMEM or 'XLoader' in clean_triple_forMEM or 'savetextfile' in clean_triple_forMEM or 'Leafminer' in clean_triple_forMEM or 'FinSpy' in clean_triple_forMEM or 'Kismet' in clean_triple_forMEM or 'Agumon' in clean_triple_forMEM or 'Gabumon' in clean_triple_forMEM or 'Biyomon' in clean_triple_forMEM or '2042' in clean_triple_forMEM or check_brackets(clean_triple_forMEM)==False or checker(triple)=='ERROR':
            print('Current short-term memory does not meet requirements',triple)
            print('Retry extracting text 2')
            triple = compute_full_extracted_triples(this_time_test)
            clean_triple_forMEM = clean_full_extracted_triples(triple)
        
        if  'Formbook' in clean_triple_forMEM or 'XLoader' in clean_triple_forMEM or 'savetextfile' in clean_triple_forMEM or 'Leafminer' in clean_triple_forMEM or 'FinSpy' in clean_triple_forMEM or 'Kismet' in clean_triple_forMEM or 'Agumon' in clean_triple_forMEM or 'Gabumon' in clean_triple_forMEM or 'Biyomon' in clean_triple_forMEM or '2042' in clean_triple_forMEM or check_brackets(clean_triple_forMEM)==False or checker(triple)=='ERROR':
            print('Current short-term memory does not meet requirements',triple)
            print('Retry extracting text 3')
            triple = compute_full_extracted_triples(this_time_test)
            clean_triple_forMEM = triple
            
        print('\nThis time short-term memory is:')
        #print("\nCLEANED TRIPLE:", clean_triple_forMEM)    
        
        if i == 0:
            if check_brackets(clean_triple_forMEM):
                longmem = clean_triple_forMEM
            else:
                longmem = 'No longterm memory'
            triple_cache.append(clean_triple_forMEM)
            text_cache.append(this_time_test)
            print('First thinking completed')
        if i >= 1:
            print('Past long-term memory is:')
            print(longmem)
            original_longmem=longmem
            if len(longmem)>=1500:
                    longmem=longmem[-1000:]
                    if '[' in longmem:
                        longmem=longmem[longmem.index('['):]
            if check_brackets(clean_triple_forMEM):
                max_retries = 3  # Maximum retry times
                retry_count = 0  # Retry counter
                while retry_count < max_retries:
                    print('Retry '+str(retry_count)+' times')
                    newlongmem = merge_extracted_triples(longmem, clean_triple_forMEM, this_time_test)
                    print('Thinking process：')
                    print(newlongmem)
                    newlongmem=newlongmem.replace('-The start of the new short-term memory area-','-The start of new short-term memory area-')
                    newlongmem=newlongmem.replace('-The end of the new short-term memory area-','-The end of new short-term memory area-') 
                    if '-The start of new short-term memory area-' in newlongmem and '-The end of new short-term memory area-' in newlongmem and checker(newlongmem)!='ERROR':
                        newlongmem=newlongmem[newlongmem.rindex('-The start of new short-term memory area-')+len('-The start of new short-term memory area-'):newlongmem.rindex('-The end of new short-term memory area-')]
                        if not any(keyword in newlongmem for keyword in ['Formbook', 'XLoader', 'savetextfile', 'Leafminer', 'FinSpy', 'Kismet','Agumon','Gabumon','Biyomon','2042']):
                            longmem = str(original_longmem)+', '+str(newlongmem)
                            retry_count=9999
                        else:
                            retry_count += 1
                    else:
                        retry_count += 1
            else:
                longmem=original_longmem
                print('Short-term memory is not a triple')
            print('\nAfter merging: The new long-term memory is:')
            print(longmem)      
            import pandas as pd

            # Create a new DataFrame
            new_data = pd.DataFrame({'single_article': [str(single_article)], 'longmem': [str(longmem),]})

            try:
                # Read the existing Excel file with explicit encoding handling
                longmem_cache = pd.read_excel('RQ2 result cache backup.xlsx', engine='openpyxl')
                # Convert any problematic strings to proper encoding
                longmem_cache = longmem_cache.map(lambda x: x.encode('utf-8', errors='ignore').decode('utf-8') if isinstance(x, str) else x)
                # Add new data to the end of existing data
                longmem_cache = pd.concat([longmem_cache, new_data], ignore_index=True)
            except FileNotFoundError:
                # If the file does not exist, use the new data directly
                longmem_cache = new_data
            except Exception as e:
                print(f"Error reading Excel file: {e}")
                # If there's any error, create new file
                longmem_cache = new_data

            # Ensure all strings are properly encoded before saving
            longmem_cache = longmem_cache.map(lambda x: x.encode('utf-8', errors='ignore').decode('utf-8') if isinstance(x, str) else x)
            
            # Save the updated data to the Excel file with explicit encoding
            with pd.ExcelWriter('RQ2 result cache backup.xlsx', engine='openpyxl') as writer:
                longmem_cache.to_excel(writer, index=False)
            # Add the result to the cache
            
    return longmem

def process_sentence(sentence, index):
    try:
        full_article_to_longmem(sentence)
        with open("done.txt", "a") as file:
            file.write(sentence + "\n")
        with open("done_index.txt", "w") as file:
            file.write(str(index))
    except Exception as e:
        with open("error.txt", "a") as file:
            file.write(sentence + "\n")
        print(f"Error：{e}")
        

if __name__ == "__main__":
    target= pd.read_csv('target.csv')

    emotet_sentences = target['string'].tolist()

    pool = ThreadPoolExecutor(max_workers=16)

    for i, sentence in enumerate(emotet_sentences):
        try:
            pool.submit(process_sentence, sentence, i)
        except Exception as e:
            print(f"Error：{e}")
            continue

    pool.shutdown()
