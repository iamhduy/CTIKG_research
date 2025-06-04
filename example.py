import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import os

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

def clean_full_extracted_triples(text):
    #remvoe all \n
    text=text.replace('\n','')
    #remove the space between ‘:" and "["
    import re
    text=re.sub(r':\s+\[',r'[',text)
    #remove the space between ‘[" and "["
    text=re.sub(r'\s+\[',r'[',text)
    #remove the space between ‘]" and "]"
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

def checker(my_string):
    promptmessage = [{
        "role": "user",
        "content":'You are a result checker. You are responsible for checking the result from other AI assistants. The AI assistant may say that \" I am sorry, but I am Chat AI model and I am not able to do the task \" or \" You should do it by yourself\" or \"I am sorry, but I am not able to do the task\". If you found those words or words with simlar meaning, you must reply me \"ERROR\", other wise, you should reply me \"OK\". Here is the result from other AI assistant: '+str(my_string)}]

    setmodel='01-ai/Yi-6B-Chat-8bits'
    api_key = "EMPTY"
    api_base = "http://localhost:8000/v1"

    client = OpenAI(api_key=api_key, base_url=api_base)
    stream = client.chat.completions.create(
        model=setmodel,
        messages=promptmessage,
        stream=True,
        max_tokens=128,
        temperature=1,
    )
    final_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            #print(chunk.choices[0].delta.content, end="")
            final_response += chunk.choices[0].delta.content  
    return final_response 

def extract_triples(text_input):
    def generate_prompt(text):
        promptmessage = [
        {
        "role": "user",
        "content": 
        ''' 
        As an AI trained in entity extraction and relationship extraction. You're an advanced AI expert, so even if I give you a complex sentence, you'll still be able to perform the relationship extraction task. The output format MUST be a dictionary where key is the source sentence and value is a list consisting of the extracted triple.
        A triple is a basic data structure used to represent knowledge graphs, which are structured semantic knowledge bases that describe concepts and their relationships in the physical world. A triple MUST has THREE elements: [Subject, Relation,  Object]. For example, "[Subject:FinSpy malware, Relation:was the final payload]"(2 elements) and "[Subject:FinSpy malware, Relation:was, Object:the final payload, None:that will be used]"(4 elements) do not contain exactly 3 elements and should be discard.The subject and the object are Noun. The relation is a relation that connects the subject and the object, and expresses how they are related. For example, [Formbook, is, malware] is a triple that describes the relationship between the malware Formbook and the concept of malware. 
        In entity extraction, you follow those rules:
        Rule 1: Only extract triples that are related to cyber attacks. If a sentence does not have any triple about cyber attacks, skip the sentence and do not print it in your output.\
        Rule 2: Make sure your results is a python dictionary format. One example is {source sentence1:[[subject1, relation1, object1],[subject2, relation2, object2]...],source sentence2:[[subject3, relation3, object3],[subject4, relation4, object4]...]} 
        Rule 3: You must use ellipsis in source sentence to save space. The output format should be  “First word Second word ... penu word last word”, For example, “The malware ... the system”.
        '''
        },
        {
        "role": "assistant",
        "content": "I got it."
        },
        {
        "role": "user",
        "content": "Here is one sentence from example article:\"Leafminer attempts to infiltrate target networks through various means of intrusion: watering hole websites, vulnerability scans of network services on the internet, and brute-force/dictionary login attempts.\""
        },
        {
        "role": "assistant",
        "content": "{Leafminer attempts ... of intrusion: watering hole websites, vulnerability scans of network services on the internet, and brute-force/dictionary login attempts:[[SUBJECT:Leafminer,RELATION:attempts to infiltrate,OBJECT:target networks],[SUBJECT:Leafminer,RELATION:use,OBJECT:watering hole websites],[SUBJECT:Leafminer,RELATION:use,OBJECT:vulnerability scans of network services on the internet],[SUBJECT:Leafminer,RELATION:use,OBJECT:brute-force],[SUBJECT:Leafminer,RELATION:use,OBJECT:dictionary login attempts]]}."
        },
        {
        "role": "user",
        "content": "Here is one sentence from example article:\"Kismet is also a powerful tool for penetration testers that need to better understand their target and perform wireless LAN discovery.\""
        },
        {
        "role": "assistant",
        "content": "{Kismet is ... wireless LAN discovery.:[[SUBJECT:Kismet,RELATION:is a powerful tool for, OBJECT:penetration testers],[SUBJECT:testers, RELATION:understand, OBJECT:their target],[SUBJECT:testers,RELATION: perform, OBJECT:wireless LAN discovery]]}."
        },
        {"role": "user",
        "content": 
        """
        \Here are my new sentence, extract all possible entity triples from it. Now, I start to give you sentence.\""
        """+text+
        """\"Now, my input text are over. You MUST follow the rules I told you before. 
        """
        },
        ]
        return promptmessage

    def generate_prompt_basedon3(inSent,inlist):
        promptmessage = [
        {
        "role": "user",
        "content":'You are responsible for combining the three different entity extraction results from three different assistants extracting from the same sentence into one. The triple is a basic data structure used to represent knowledge graphs, which are structured semantic knowledge bases that describe concepts and their relationships in the world. A triple MUST have THREE elements: [subject, relation, object]. The subject has the prefix "SUBJECT:",the relation has prefix "RELATION:", the object has prefix "OBJECT:", some triples examples are "[SUBJECT:The user, RELATION:logs in, OBJECT:the system],[SUBJECT:The system, RELATION:stores, personal information],[SUBJECT:The system, RELATION:sends, OBJECT:personal information]". The final results is a python dictionary format. One example of result is {source sentence1:[[subject1, relation1, object1],[subject2, relation2, object2]...]}. Some assistants use ellipses to simplify words source sentence, for example "The exploit was delivered through a Microsoft Office document and the final payload was the latest version of FinSpy malware." and "The exploit ... FinSpy malware." and "The exploit was delivered ... latest version of FinSpy malware." are the same one sentence. So when you find the different dictionary key that has same beginning and ending words, you should combine them into one dict. I would like you to integrate these three results into one and discard the exact same triples and discard triples that do not contain exactly 3 elements, for example "[SUBJECT:FinSpy malware, RELATION:was the final payload]"(2 elements) and "[SUBJECT:FinSpy malware, RELATION:SUBJECT:was, OBJECT:the final payload, UNKNOWN:that will be used]"(4 elements) do not contain exactly 3 elements and should be discard. The source sentence is '+str(inSent)+', the extracted triples result are'+str(inlist)+'Just answer me the final python dictionary with triple format without any other words.'
        },
        ]
        return promptmessage
    
    def generate_prompt_postprocess(text):
        promptmessage = [
        {
        "role": "user",
        "content": 
        ''' 
        You play the role of an entity extraction expert and modify/simplify/split the text (extracted multiple triples) in the entity extraction result I gave you (a python dictionary with key as the source sentence with ellipsis and value as the extracted triples) according to the following rules. A triple is a basic data structure used to represent knowledge graphs, which are structured semantic knowledge bases that describe concepts and their relationships in the physical world. A triple consists of three elements: [SUBJECT, RELATION,OBJECT]. The subject and the object are entities, which can be things, people, places, events, or abstract concepts. The relation is a relation that connects the subject and the object, and expresses how they are related. For example, [Formbook, is, malware] is a triple that describes the relationship between the malware Formbook and the concept of malware.
        Rule 1: If the subject or object in a triple contains pronouns such as it, they, malware, Trojan, attack, ransomware, or group, replace them with a specific name as much as possible according to the context, such as “CVE-xxx” or “XLoader” will replace "it" or "malware" if context has this relationship information.
        Rule 2: Focus on malware, Trojan horse, CVE, or hacking organization as the subject of the triples, if a subject with "malware" or "Trojan horse" or "CVE" or "hacking organization" is found and has additional suffixes, remove the suffixes.
        Rule 3: Split a complex triple into multiple simpler forms. For example, [Formbook and XLoader, are,malware] should be split into [Formbook,is,malware] and [XLoader,is,malware].
        Rule 4: If the [subject,relation] in a triple can be formed into a new [subject,relation,object] triple because relation itself has a new object in it, create a new triple while keeping the original one. 
        Rule 5: If the object can be simplified to a more concise, generic expression, create a new triple while keeping the original one. For example, ["Formbook", "save", "XLoader in desktop"] MUST has a new triple ["Formbook", "save", "XLoader"] due to the object "XLoader in desktop" can be simplified to "XLoader".
        Rule 6: Simplify the subject, object, and relation into a more concise, generic expression.
        Rule 7:When you encounter a subject or object that contains modifiers and adjectives, remove them. For example, [a notorious Formbook malware] should be simplified to [Formbook].
        Rule 8:When you encounter a plural or past tense form, convert it to singular or present tense. For example, [Windows users] should be converted to [Windows user].
        Rule 9:When you encounter an MD5, registry, path, or other identifier that contains prefixes, remove them. For example, [md5 xxxxx] should be simplified to [xxxxx].
        Rule 10:When you encounter a proper noun that contains a suffix, remove the suffix. For example, [“Specific names of a malware/ransomware/trojan” malware/ransomware/trojan] should be simplified to [“Specific names of a malware/ransomware/trojan”]
        Rule 11: Make sure the subject has a prefix "SUBJECT:", the relation has prefix "RELATION:", the object has prefix "OBJECT:", a triple example is "[SUBJECT:Formbook, RELATION:save, OBJECT:a file]
        '''
        +"Here is my entity extraction result:"+str(text)+"Now, you apply the rules I told you before. Write down your though, think it step by step. If all triple don't need to be modified based on specific rule, just write down 'no change'.In the end, you MUST tell me the final new entity extraction result. Make sure your results contain a dictionary where key is the original sentence and value is a list consisting of the extracted triple for subsequent information extraction."
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

    def api_call(text, temp):
        from openai import OpenAI

        setmodel="01-ai/Yi-6B-Chat-8bits"
        api_key = "EMPTY"
        api_base = "http://localhost:8000/v1"
        promptmessage = text

        client = OpenAI(api_key=api_key, base_url=api_base)

        token = 1024
        while True:
            if token == 0:
                return "ERROR"
            
            try:
                stream = client.chat.completions.create(
                model=setmodel,
                messages=promptmessage,
                stream=True,
                max_tokens=token,
                temperature=temp,
                )
                break
            except Exception as e:
                print("Exception:", e)
                token = token // 2
                stream = client.chat.completions.create(
                model=setmodel,
                messages=promptmessage,
                stream=True,
                max_tokens=token,
                temperature=temp,
            )

        final_response = ""

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                #print(chunk.choices[0].delta.content, end="")
                final_response += chunk.choices[0].delta.content  

        return final_response

    #triples_list = []
    response_list = []
    temp_list = [1, 0.7, 0.5]
    for temp in temp_list:
        response = api_call(generate_prompt(text_input), temp)
        #print('Response:\n', response)
        #clean_response = get_only_triples(response)
        cleaned_text = clean_text(str(response))
        
        if any(keyword in cleaned_text for keyword in ['CVExxx', 'Formbook', 'XLoader', 'Malwaresavetextfile', 'Leafminer', 'FinSpy', 'Kismet', 'Specificnamesofa']):
            response_list.append('ERROR')
        else:
            response_list.append(get_only_triples(response))

    #print('--------------------------------')

    group_response = api_call(generate_prompt_basedon3(text_input, response_list), 0.7)
    #print('Group:\n', group_response)
    #print('--------------------------------')
    refined_response = api_call(generate_prompt_postprocess(get_only_triples(group_response)), 0.7)
    #print('Refined:\n', refined_response)
    #print('--------------------------------')
    cleaned_response = get_only_triples(refined_response)
    #cleaned_response = cleaned_response.replace(' ', '')
    #print('cleaned_response:\n', cleaned_response)
    return cleaned_response

def merge_triples(longmem, shortmem):
    def api_call(text, temp=0.7):
        from openai import OpenAI

        setmodel="01-ai/Yi-6B-Chat-8bits"
        api_key = "EMPTY"
        api_base = "http://localhost:8000/v1"
        promptmessage = text
        promp_len = 0
        for i in promptmessage:
            promp_len += len(i['content'])

        client = OpenAI(api_key=api_key, base_url=api_base)

        token = 1024
        while True:
            if token == 0:
                return "ERROR"
            
            try:
                stream = client.chat.completions.create(
                model=setmodel,
                messages=promptmessage,
                stream=True,
                max_tokens=token,
                temperature=temp,
                )
                break
            except Exception as e:
                print("Exception:", e)
                token = token // 2
                stream = client.chat.completions.create(
                model=setmodel,
                messages=promptmessage,
                stream=True,
                max_tokens=token,
                temperature=temp,
            )

        final_response = ""

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                #print(chunk.choices[0].delta.content, end="")
                final_response += chunk.choices[0].delta.content  
        return final_response
    
    promptmessage = [{"role": "user",
        "content": '''You are a triples integration assistant. Your task is to extract and integrate triples (Subject, Relation, Object) from article parts while maintaining consistency.
Memory Areas:
1. Long-term memory: Stores triples from previously read article parts
2. Short-term memory: Stores triples from current article part
3. New short-term memory: Modified triples after integration

Integration Rules:
1. Unify entities: Remove prefixes/suffixes/modifiers from identical terms
   Example: [the Formbook, is, malware] → [Formbook, is, malware]

2. Preserve specific names: Keep exact names for malware, CVEs, Trojans, organizations
   Example: [Malware Formbook, is, malware] → [Formbook, is, malware]

3. No inference: Don't create new triples without direct evidence
   Example: If long-term has [malware, downloads, X] and short-term has [Formbook, is, malware],
   don't create [Formbook, downloads, X]

4. No creation: Only use triples from existing memories

5. No examples: Don't include example words in output

Format:
- Each triple must be [Subject, Relation, Object]
- New short-term memory must be enclosed in:
  -The start of new short-term memory area-
  [triples here]
  -The end of new short-term memory area-

Process:
1. Review long-term and short-term memories
2. Apply rules to modify triples
3. Document rule applications
4. Create new short-term memory with modified triples

-The start of the long-term memory area-
'''+str(longmem)+'''
-The end of the long-term memory area-
    
-The start of the short-term memory area-
'''+str(shortmem)+'''
-The end of the short-term memory area-
        
Now, follow the rules. Write down how you use the rule to modify the triples in short-term memory. Then, write down new short-term memory which must be started with \'-The start of new short-term memory area-\' and ended with \'-The end of new short-term memory area-\'
'''},
]
    return api_call(promptmessage)

def longmem_clean(longmem):
    response = longmem.replace('SUBJECT:,', '').replace('RELATION:,', '').replace('OBJECT:,', '')
    response = response.replace('SUBJECT:', '').replace('RELATION:', '').replace('OBJECT:', '')
    response = response.replace('SUBJECT', '').replace('RELATION', '').replace('OBJECT', '')

    while '  ' in response:
        response = response.replace('  ',' ')

    while '[ ' in response:
        response = response.replace('[ ','[')

    while ' ]' in response:
        response = response.replace(' ]',']')

    return response

def article_to_longmem(single_article):
        grouped_texts_strings = full_text_to_parts(single_article)
        triple_cache = []
        text_cache = []
        for i in range(len(grouped_texts_strings)):
            this_time_test=grouped_texts_strings[i]
            if len(this_time_test) > 1500:
                this_time_test = this_time_test[0:1500]
            #print('Thinking about paragraph '+str(i))
            print('\nSeeing text：')
            print(this_time_test)
            triple = extract_triples(this_time_test)
            clean_triple_forMEM = clean_full_extracted_triples(triple)
            #print('triple:',triple)

            if  'Formbook' in clean_triple_forMEM or 'XLoader' in clean_triple_forMEM or 'savetextfile' in clean_triple_forMEM or 'Leafminer' in clean_triple_forMEM or 'FinSpy' in clean_triple_forMEM or 'Kismet' in clean_triple_forMEM or 'Agumon' in clean_triple_forMEM or 'Gabumon' in clean_triple_forMEM or 'Biyomon' in clean_triple_forMEM or '2042' in clean_triple_forMEM or check_brackets(clean_triple_forMEM)==False or checker(triple)=='ERROR':
                print('Current short-term memory does not meet requirements',triple)
                print('Retry extracting text 1')
                triple = extract_triples(this_time_test)
                clean_triple_forMEM = clean_full_extracted_triples(triple)

            if  'Formbook' in clean_triple_forMEM or 'XLoader' in clean_triple_forMEM or 'savetextfile' in clean_triple_forMEM or 'Leafminer' in clean_triple_forMEM or 'FinSpy' in clean_triple_forMEM or 'Kismet' in clean_triple_forMEM or 'Agumon' in clean_triple_forMEM or 'Gabumon' in clean_triple_forMEM or 'Biyomon' in clean_triple_forMEM or '2042' in clean_triple_forMEM or check_brackets(clean_triple_forMEM)==False or checker(triple)=='ERROR':
                print('Current short-term memory does not meet requirements',triple)
                print('Retry extracting text 2')
                triple = extract_triples(this_time_test)
                clean_triple_forMEM = clean_full_extracted_triples(triple)
            
            if  'Formbook' in clean_triple_forMEM or 'XLoader' in clean_triple_forMEM or 'savetextfile' in clean_triple_forMEM or 'Leafminer' in clean_triple_forMEM or 'FinSpy' in clean_triple_forMEM or 'Kismet' in clean_triple_forMEM or 'Agumon' in clean_triple_forMEM or 'Gabumon' in clean_triple_forMEM or 'Biyomon' in clean_triple_forMEM or '2042' in clean_triple_forMEM or check_brackets(clean_triple_forMEM)==False or checker(triple)=='ERROR':
                print('Current short-term memory does not meet requirements',triple)
                print('Retry extracting text 3')
                triple = extract_triples(this_time_test)
                clean_triple_forMEM = triple
                
            print('This time short-term memory is:')
            while '  ' in clean_triple_forMEM:
                clean_triple_forMEM = clean_triple_forMEM.replace('  ',' ')
            print(clean_triple_forMEM)

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
                        newlongmem = merge_triples(longmem, clean_triple_forMEM)
                        print('Thinking process：')
                        print(newlongmem)
                        newlongmem=newlongmem.replace('-The start of the new short-term memory area-','-The start of new short-term memory area-')
                        newlongmem=newlongmem.replace('-The end of the new short-term memory area-','-The end of new short-term memory area-') 
                        if '-The start of new short-term memory area-' in newlongmem and '-The end of new short-term memory area-' in newlongmem:
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
                print('After merging: The new long-term memory is:')
                print(longmem)      
                # Create a new DataFrame
                new_data = pd.DataFrame({'single_article': [str(single_article)], 'longmem': [str(longmem),]})

                try:
                    # Read the existing Excel file
                    longmem_cache = pd.read_excel('RQ2 result cache backup.xlsx')
                    # Add new data to the end of existing data
                    longmem_cache = pd.concat([longmem_cache, new_data], ignore_index=True)
                except FileNotFoundError:
                    # If the file does not exist, use the new data directly
                    longmem_cache = new_data

                # Save the updated data to the Excel file
                longmem_cache.to_excel('RQ2 result cache backup.xlsx', index=False)

        new_data = pd.DataFrame({'single_article': [str(single_article)], 'longmem': [str(longmem_clean(longmem))]})
        return new_data

if __name__ == "__main__":
    try:
        os.remove('triples_list.txt')
        os.remove('RQ2 result cache backup.xlsx')
    except:
        print('No file to remove')

    df = pd.read_csv('target.csv')
    size = len(df['string'])
    pool = ThreadPoolExecutor(max_workers=16)
    longmem_list = []

    for j in range(size):
        print('Sentence', j)
        article = df['string'][j]
        try:
            longmem_list.append(article_to_longmem(article))
        except Exception as e:
            data = pd.DataFrame({'single_article': [str(j)], 'longmem': [str(e)]})
            longmem_list.append(data)
    longmem_pd = pd.concat(longmem_list, ignore_index=True)
    longmem_pd.to_excel('result.xlsx', index=False)



    

