# Import necessary libraries

# Streamlit is a Python library for creating web applications with minimal effort.
import streamlit as st

# PyPDF2 is a library for working with PDF files, such as reading and manipulating their content.
import PyPDF2

# BytesIO is a class from the io module that provides a way to work with in-memory binary data as if it were a file.
from io import BytesIO

# OpenAI is a platform that provides powerful natural language processing APIs. (Note: It seems like the 'openai' library is imported but not used in the provided code snippet.)
import openai

# pydub is a library for audio file manipulation (Note: It seems like the 'pydub' library is imported but not used in the provided code snippet.)
import pydub

# requests is a library for making HTTP requests.
import requests

# json is a standard library for working with JSON data.
import json

# time is a standard library for working with time-related functions.
import time

# re is a regular expression library for pattern matching in strings.
import re

# Set Streamlit page configuration
st.set_page_config(
    page_title="NoteMaker",
    page_icon=":bookmark_tabs:",
)

# Create sidebar with useful prompt
st.sidebar.markdown("App made by Arun :heart:")
st.sidebar.markdown("# Useful Prompts")

st.sidebar.markdown("### General Note Making")
st.sidebar.markdown("- Generate detailed call notes of the conversation for an investment firm.\nGenerate notes pointwise with full sentences under each of all the Important Sections, (Convert all text numbers to numbers, Include all important information and numbers.)\n")

st.sidebar.markdown("### Custom Topic Input")
st.sidebar.markdown("- Generate detailed call notes of the conversation for an investment firm only under this topic-{'<topic>'}. Do not create any subtopics under the topics. Generate notes pointwise under only that topic, (Convert all text numbers to numbers)")

st.sidebar.markdown("### Misc")
st.sidebar.markdown("- Please revise the following text by eliminating any repetitive information and clubbing similar topics under common headings. Ensure that each heading is followed by its relevant points. Do not rewrite any point or make them smaller. Avoid using bold formatting and numbering for the headings.")

# Create sidebar with useful prompt
st.markdown('# Hello User!')

# User selects the source type (pdf, audio, or gdrive link) using a dropdown select box.
file_type = st.selectbox(
    'Choose source type [pdf, audio, audiogest-link]:',
    ('pdf', 'audio', 'gdrive link(public access)')
)

# If the selected source type is 'gdrive link(public access)', user is prompted to enter the Google Drive public access link.
link_input = st.text_input('Enter the google drive public access link of the file here(if you have chosen "gdrive link(public access)" in the previous question):')
link_input_value = ''
if link_input:
    link_input_value = link_input

# Settings for Audiogest/Whisper are provided in an expandable section.
with st.expander('Whisper/Audiogest Settings', expanded=False):
    # User is prompted to enter the language of the audio in ISO-639-1 format.
    language_input = st.text_input('Enter the language of the audio in "ISO-639-1" format {english = en, hindi = hi}:', value='en')
    
    # User is prompted to enter a custom prompt containing factual words present in the audio.
    prompt_input = st.text_area('Enter your custom prompt which may contain factual words present in the audio:')
    
    # User is prompted to enter a number between 0 and 1 for the temperature value.
    temperature_input = st.number_input("Enter a number between [0,1] for temperature value:", value=0.3)
    
    # User is prompted to enter the number of speakers, including the interviewer (for audiogest).
    num_speakers_input = st.number_input("Enter the number of speakers including the interviewer(for audiogest):", value=2)
    
    # User is prompted to enter the time to wait for audiogest transcription (in minutes).
    wait_time_input = st.number_input("Enter the time(in minutes) to wait for audiogest transcription:", value=10)

# Setting default values for the inputs in case the user does not provide any value.
language_input_value='en'
if language_input:
    language_input_value = language_input

prompt_input_value=''
if prompt_input:
    prompt_input_value = prompt_input

temperature_input_value=0.3
if temperature_input:
    temperature_input_value = temperature_input

num_speakers_input_value=2
if num_speakers_input:
    num_speakers_input_value = num_speakers_input

wait_time_input_value=10
if wait_time_input:
    wait_time_input_value = wait_time_input


# GPT parameters are provided in an expandable section.
with st.expander('GPT parameters', expanded=False):
    # User is prompted to enter a number between 0 and 2 for the temperature value of GPT Note Making.
    temperature_input_GPT = st.number_input("Enter a number between [0,2] for temperature value of GPT Note Making:", value=1.5)
    
    # User is prompted to enter the 'Top p' input value for GPT Note Making.
    top_p_input_GPT = st.number_input("Enter the 'Top p' input value of GPT Note Making:", value=0.6)
    
    # User is prompted to enter the 'frequency_penalty' input value for GPT Note Making.
    frequency_penalty_GPT = st.number_input("Enter the 'frequency_penalty' input value of GPT Note Making:", value=0.1)
    
    # User is prompted to enter the 'presence_penalty' input value for GPT Note Making.
    presence_penalty_GPT = st.number_input("Enter the 'presence_penalty' input value of GPT Note Making:", value=0.1)
    
    # User can optionally upload a context PDF file.
    context_file = st.file_uploader("[optional] Context PDF file:", type=["pdf"], accept_multiple_files=False)
    
    # User selects the GPT model to use from a dropdown select box.
    model_option = st.selectbox(
        'Which model would you like to use?',
        ('gpt-4-1106-preview', 'gpt-4', 'gpt-3.5-turbo-1106')
    )

# Setting default values for GPT parameters in case the user does not provide any value.
temperature_input_GPT_value=1.5
if temperature_input_GPT:
    temperature_input_GPT_value = temperature_input_GPT

top_p_input_GPT_value=0.6
if top_p_input_GPT:
    top_p_input_GPT_value = top_p_input_GPT

frequency_penalty_GPT_value=0.1
if frequency_penalty_GPT:
    frequency_penalty_GPT_value = frequency_penalty_GPT

presence_penalty_GPT_value=0.1
if presence_penalty_GPT:
    presence_penalty_GPT_value = presence_penalty_GPT

# User is prompted to enter the chunk size (max_len) for processing.
max_len_str = st.text_input('Chunk size :red[[required]] :')
if max_len_str:
    max_len = int(max_len_str)

# User is prompted to enter a title for the file.
file_title = st.text_input('File title:')
if file_title:
    pass
else:
    # If no file title is provided, a placeholder is used.
    file_title = 'place_holder_for_file_name'

# If a context file is uploaded, its contents are extracted for further processing.
if context_file:
    context_file_contents = ''
    
    # Using PyPDF2 to read the content of the PDF file page by page.
    reader = PyPDF2.PdfReader(context_file)
    
    # Loop through each page and extract text content.
    for j in range(len(reader.pages)):
        p = reader.pages[j]
        t = p.extract_text()
        
        # Append the extracted text to the context_file_contents.
        context_file_contents = context_file_contents + "\n" + t

else:
    # If no context file is uploaded, an empty string is assigned to context_file_contents.
    context_file_contents = ''


operation_option = st.selectbox(
    'Which operation do you want to perform?',
    ('General Note Making', 'Custom Topic Input'))

topic_input_file = st.file_uploader("Choose a :red[PDF file containing the topics] arranged properly:", type=["pdf"], accept_multiple_files=False)

user_prompt_input = st.text_input('Enter the comma seperated topics in 1 line (If you have chosen "custom topic input"):')

# If a topic input file is provided, read its content and extract topics from it.
if topic_input_file:
    input_text = ''
    
    # Using PyPDF2 to read the content of the PDF file page by page.
    reader = PyPDF2.PdfReader(topic_input_file)
    
    # Loop through each page and extract text content.
    for j in range(len(reader.pages)):
        p = reader.pages[j]
        t = p.extract_text()
        
        # Append the extracted text to the input_text.
        input_text = input_text + "\n" + t

    # Split the input text into a list of topics, removing extra spaces and newlines.
    topics_input_list = [i.strip().replace('\n','') for i in input_text.split('\n \n')]

# If user provides a custom prompt directly, split it into a list of topics.
elif user_prompt_input:
    topics_input_list = [i.strip() for i in user_prompt_input.split(',')]

# Default prompt text for the prompt area.
prompt_area_default_text = "Generate detailed call notes of the conversation for an investment firm.\nGenerate notes pointwise with full sentences under each of all the Important Sections, (Convert all text numbers to numbers, Include all important information and numbers.)\n"

# Initialize prompt_area_text with the default prompt.
prompt_area_text=''

# If an operation_option is provided, update the prompt_area_default_text based on the selected operation.
if operation_option:
    if operation_option=='General Note Making':
        prompt_area_default_text = "Generate detailed call notes of the conversation for an investment firm.\nGenerate notes pointwise with full sentences under each of all the Important Sections, (Convert all text numbers to numbers, Include all important information and numbers.)\n"
    elif operation_option=='Custom Topic Input':
        prompt_area_default_text = "Generate detailed call notes of the conversation for an investment firm only under this topic-{'<topic>'}. Do not create any subtopics under the topics. Generate notes pointwise under only that topic, (Convert all text numbers to numbers)"

# Set prompt_area_text to the default or updated prompt text.
prompt_area_text = st.text_area("This is the default prompt. If you modify the prompt, please choose 'Use customized prompt' in the next question_____ :red[DO NOT REMOVE THE <topic> PART OF THE PROMPT IN Custom Topic Input]", value=prompt_area_default_text)

# User selects whether to use the default prompt or a customized prompt.
prompt_option = st.selectbox(
    ':red[Do you want to use the customized prompt ?]',
    ('Use default prompt', 'Use customized prompt'))

# User uploads a PDF/Audio file.
uploaded_file = st.file_uploader("Choose a PDF/Audio file:", type=["pdf","mp3","mp4","m4a","wav"], accept_multiple_files=True)



# Function for generating notes using OpenAI GPT based on user input and parameters.
def Note_maker(model_option, t_list, api_key, prompt_option, prompt_area_text, context_file_contents):
    # Initialize OpenAI API client with the provided API key.
    client = openai.OpenAI(api_key=api_key)
    
    # Display progress update in the Streamlit app.
    st.write('[Note Making] Progress update:','\n')
    st.write(1,'/',len(t_list),'\n')

    # Determine the actual prompt based on user's choice.
    if prompt_option == "Use default prompt":
        actual_prompt = "Generate detailed call notes of the conversation for an investment firm.\nGenerate notes pointwise with full sentences under each of all the Important Sections, (Convert all text numbers to numbers, Include all important information and numbers.)\n"
    elif prompt_option == "Use customized prompt":
        actual_prompt = prompt_area_text

    # Initialize a list of messages with system and user prompts for the first topic in t_list.
    message_list = [
        {
          "role": "system",
          "content": actual_prompt
        },
        {
          "role": "user",
          "content": t_list[0]
        }
    ]
    
    # Create GPT completions for the first topic.
    response = client.chat.completions.create(
        model=model_option,
        messages=message_list,
        temperature=temperature_input_GPT_value,
        max_tokens=4096,
        top_p=top_p_input_GPT_value,
        frequency_penalty=frequency_penalty_GPT_value,
        presence_penalty=presence_penalty_GPT_value
    )
    
    # Initialize an empty list to store notes.
    Notes = []
    
    # Loop through the rest of the topics in t_list.
    for i in range(1, len(t_list)):
        st.write(i + 1, '/', len(t_list), '\n')
        
        # Append the generated note for the current topic to the Notes list.
        Notes.append(response.choices[0].message.content)

        # Update the message_list for the next GPT completion.
        if (model_option == 'gpt-3.5-turbo-1106') or (model_option == 'gpt-4'):
            try:
                del message_list[2:4]
            except:
                pass
        elif (model_option == 'gpt-4-1106-preview'):
            try:
                if len(message_list) >= 8:
                    del message_list[2:4]
            except:
                pass

        # Append assistant's response to message_list, including context_file_contents and user's input for the next topic.
        message_list.append({
            "role": "assistant",
            "content": context_file_contents + response.choices[0].message.content
        })
        
        # Prepare the continuation of the transcript for the next topic.
        stock = "This is continuation of the transcript.\nGenerate call notes pointwise for an investment firm under each of all the Important Sections, (Convert all text numbers to numbers, Include all important information and numbers.)\n"
        cont = stock + t_list[i]
        
        # Append user's input for the next topic to message_list.
        message_list.append({
            "role": "user",
            "content": cont
        })
        
        # Create GPT completions for the next topic.
        response = client.chat.completions.create(
            model=model_option,
            messages=message_list,
            temperature=temperature_input_GPT_value,
            max_tokens=4096,
            top_p=top_p_input_GPT_value,
            frequency_penalty=frequency_penalty_GPT_value,
            presence_penalty=presence_penalty_GPT_value
        )
    
    # Append the note for the last topic to the Notes list.
    Notes.append(response.choices[0].message.content)

    # Concatenate all the generated notes into a single string.
    Notes_Final = ''
    for i in Notes:
        Notes_Final = Notes_Final + i + '\n\n'

    # Display completion message in the Streamlit app.
    st.write('Process done!','\n')
    
    # Return the final notes.
    return Notes_Final

# An Experiment
# def Custom_Note_maker(model_option, t_list, api_key, user_prompt_input, prompt_option, prompt_area_text):
# 	client = openai.OpenAI(api_key=api_key)
# 	st.write('[Custom Input Note Making] Progress update:','\n')
# 	Notes = []
	
# 	for i in range(len(t_list)):
# 		st.write(i+1,'/',len(t_list),'\n')

# 		if prompt_option=="Use default prompt":
# 			actual_prompt = "Generate detailed call notes of the conversation for an investment firm only under these topics-{"+user_prompt_input+"}.\nGenerate notes pointwise under only those topics, (Convert all text numbers to numbers)"
# 		elif prompt_option=="Use customized prompt":
# 			actual_prompt = prompt_area_text
		
# 		message_list = [
# 		{
# 		  "role": "system",
# 		  "content": actual_prompt
# 		},
# 		{
# 		  "role": "user",
# 		  "content": 'Notes: \n'+t_list[i]
# 		}
# 		]
		
# 		response = client.chat.completions.create(
# 		model='gpt-4-1106-preview',
# 		messages=message_list,
# 		temperature=temperature_input_GPT_value,
# 		max_tokens=4096,
# 		top_p=top_p_input_GPT_value,
# 		frequency_penalty=frequency_penalty_GPT_value,
# 		presence_penalty=presence_penalty_GPT_value
# 		)
		
# 		Notes.append(response.choices[0].message.content)

# 	topics = [i.strip() for i in user_prompt_input.split(',')]
	
# 	topics_names = [s.replace(' ', '_') for s in topics]
	
# 	for j in range(len(topics)):
# 		globals()[topics_names[j]+"_items"] = ''
	
# 	for i in range(len(Notes)):
# 		ex = [i.strip() for i in Notes[i].strip().split('\n\n')]
# 		if len(ex)>=len(topics):
# 			for j in range(len(topics)):
# 				globals()[topics_names[j]+"_items"] = globals()[topics_names[j]+"_items"] + ex[j]
		    
# 	Custom_notes = ''
	
# 	for j in range(len(topics)):
# 		Custom_notes = Custom_notes + globals()[topics_names[j]+"_items"] + '\n\n'

# 	return Custom_notes

# Function for generating notes for multiple uploaded files using OpenAI GPT based on user input and parameters.
def Multi_Note_maker(uploaded_file, model_option, t_list, api_key, prompt_option, prompt_area_text, context_file_contents):
    # Initialize OpenAI API client with the provided API key.
    client = openai.OpenAI(api_key=api_key)
    
    # Initialize an empty string to store the final notes for all uploaded files.
    Notes_Final_Final = ''
    
    # Loop through each set of topics associated with each uploaded file.
    for j in range(len(t_list)):
        # Display progress update for each set of topics in the Streamlit app.
        st.write(f'[Note Making {j+1}] Progress update:','\n')
        st.write(1,'/',len(t_list[j]),'\n')

        # Determine the actual prompt based on user's choice.
        if prompt_option == "Use default prompt":
            actual_prompt = "Generate detailed call notes of the conversation for an investment firm.\nGenerate notes pointwise with full sentences under each of all the Important Sections, (Convert all text numbers to numbers, Include all important information and numbers.)\n"
        elif prompt_option == "Use customized prompt":
            actual_prompt = prompt_area_text

        # Initialize a list of messages with system and user prompts for the first topic in t_list[j].
        message_list = [
            {
            "role": "system",
            "content": actual_prompt
            },
            {
            "role": "user",
            "content": t_list[j][0]
            }
        ]
        
        # Create GPT completions for the first topic in the current set.
        response = client.chat.completions.create(
            model=model_option,
            messages=message_list,
            temperature=temperature_input_GPT_value,
            max_tokens=4096,
            top_p=top_p_input_GPT_value,
            frequency_penalty=frequency_penalty_GPT_value,
            presence_penalty=presence_penalty_GPT_value
        )
        
        # Initialize an empty list to store notes for the current set of topics.
        Notes = []
        
        # Loop through the rest of the topics in t_list[j].
        for i in range(1, len(t_list[j])):
            st.write(i+1,'/',len(t_list[j]),'\n')
            
            # Append the generated note for the current topic to the Notes list.
            Notes.append(response.choices[0].message.content)

            # Update the message_list for the next GPT completion.
            if (model_option == 'gpt-3.5-turbo-1106') or (model_option == 'gpt-4'):
                try:
                    del message_list[2:4]
                except:
                    pass
            elif (model_option == 'gpt-4-1106-preview'):
                try:
                    if len(message_list) >= 8:
                        del message_list[2:4]
                except:
                    pass

            # Append assistant's response to message_list, including context_file_contents and user's input for the next topic.
            message_list.append({
                "role": "assistant",
                "content": context_file_contents + response.choices[0].message.content
            })
            
            # Prepare the continuation of the transcript for the next topic.
            stock = "This is continuation of the transcript.\nGenerate call notes pointwise for an investment firm under each of all the Important Sections, (Convert all text numbers to numbers, Include all important information and numbers.)\n"
            cont = stock + t_list[j][i]
            
            # Append user's input for the next topic to message_list.
            message_list.append({
                "role": "user",
                "content": cont
            })
            
            # Create GPT completions for the next topic in the current set.
            response = client.chat.completions.create(
                model=model_option,
                messages=message_list,
                temperature=temperature_input_GPT_value,
                max_tokens=4096,
                top_p=top_p_input_GPT_value,
                frequency_penalty=frequency_penalty_GPT_value,
                presence_penalty=presence_penalty_GPT_value
            )
        
        # Append the note for the last topic in the current set to the Notes list.
        Notes.append(response.choices[0].message.content)

        # Concatenate all the generated notes for the current set into a single string.
        Notes_Final = ''
        for i in Notes:
            Notes_Final = Notes_Final + i + '\n\n'

        # Concatenate the notes for the current set with the file name and add to the final string.
        Notes_Final_Final = Notes_Final_Final + uploaded_file[j].name + ' :\n' + Notes_Final + '\n\n\n'
        
    # Display completion message in the Streamlit app.
    st.write('Process done!','\n')
    
    # Return the final notes for all uploaded files.
    return Notes_Final_Final

# Function for generating custom notes for multiple uploaded files using OpenAI GPT based on user input and parameters.
def Multi_Custom_Note_maker(uploaded_file, model_option, full_text, api_key, topics_input_list, prompt_option, prompt_area_text):
    # Initialize OpenAI API client with the provided API key.
    client = openai.OpenAI(api_key=api_key)
    
    # Initialize an empty string to store the final custom notes.
    Notes_Final_Final = '' 
    
    # Extract topics from the user-provided list.
    topics = topics_input_list

    # Determine the actual prompt based on user's choice.
    if prompt_option == "Use default prompt":
        actual_prompt = "Generate detailed call notes of the conversation for an investment firm only under this topic-{'<topic>'}. Do not create any subtopics under the topics. Generate notes pointwise under only that topic, (Convert all text numbers to numbers)"
    elif prompt_option == "Use customized prompt":
        actual_prompt = prompt_area_text

    # Loop through each topic in the list.
    for i in range(len(topics)):
        st.write(f"[Custom Input Note Making {i+1}/{len(topics)}] Progress update:",'\n')
        
        # Append the current topic to the final string.
        Notes_Final_Final = Notes_Final_Final + "     " + topics[i] + ':\n'

        # Loop through each uploaded file's content.
        for j in range(len(full_text)):   
            st.write(j+1,'/',len(full_text),'\n')
            
            # Initialize a list of messages with system and user prompts for the current topic and file content.
            message_list = [
                {
                "role": "system",
                "content": actual_prompt.replace('<topic>', topics[i])
                },
                {
                "role": "user",
                "content": 'Notes: \n'+full_text[j]
                }
            ]

            # Create GPT completions for the current topic and file content.
            response = client.chat.completions.create(
                model='gpt-4-1106-preview',
                messages=message_list,
                temperature=temperature_input_GPT_value,
                max_tokens=4096,
                top_p=top_p_input_GPT_value,
                frequency_penalty=frequency_penalty_GPT_value,
                presence_penalty=presence_penalty_GPT_value
            )
            
            # Append the file name and GPT response content to the final string.
            Notes_Final_Final = Notes_Final_Final + "         " + uploaded_file[j].name + ":\n"
            Notes_Final_Final = Notes_Final_Final + response.choices[0].message.content + "\n\n"
        
        # Add extra line breaks to separate each topic's notes.
        Notes_Final_Final = Notes_Final_Final + "\n\n\n\n"

    # Return the final custom notes.
    return Notes_Final_Final

# Function for processing uploaded PDF files.
def pdf_processor(uploaded_file, max_len):
    # Initialize a list to store the full text for each uploaded PDF file.
    full_text = ["" for k in range(len(uploaded_file))]
    
    # Loop through each uploaded PDF file.
    for i in range(len(uploaded_file)):
        # Read the content of each page in the PDF file and concatenate it.
        reader = PyPDF2.PdfReader(uploaded_file[i])
        for j in range(len(reader.pages)):
            p = reader.pages[j]
            t = p.extract_text()
            full_text[i] = full_text[i] + "\n" + t
    
    # Store the full text in Transcript_final variable.
    Transcript_final = full_text
    
    # Initialize a list to store segmented text for each file.
    t_list = [[] for k in range(len(Transcript_final))]
    
    # Determine the number of words per segment based on the specified max_len.
    words_per_segment = max_len
    
    # Loop through each file's transcript and segment the text.
    for j in range(len(Transcript_final)):
        words = Transcript_final[j].split()
        
        for i in range(0, len(words), words_per_segment):
            segment = " ".join(words[i:i + words_per_segment])
            t_list[j].append(segment)
    
    # Return the segmented text and full text.
    return t_list, full_text

# Function for processing uploaded audio files using the Whisper model.
def audio_processor_whisper(uploaded_file, max_len, string_transcript_audio, language_input_value, prompt_input_value, temperature_input_value):
    # Load the audio file from the uploaded file.
    audio = pydub.AudioSegment.from_file(uploaded_file)
    
    # Calculate the total duration of the audio.
    total_duration = len(audio)
    
    # Define the chunk length in milliseconds.
    chunk_length_ms = 60000
    
    # Calculate the number of chunks needed for the entire audio file.
    num_chunks = total_duration // chunk_length_ms
    
    # Initialize OpenAI API client with the provided API key.
    client = openai.OpenAI(api_key=st.secrets["openai_key"])
    
    # Display progress update for transcription in the Streamlit app.
    st.write('[Transcription] Progress update:','\n')
    
    # Loop through each chunk of the audio file for transcription.
    for i in range(num_chunks):
        st.write(i+1,'/',num_chunks,'\n')
        
        # Define the start and end time for the current chunk.
        start_time = i * chunk_length_ms
        end_time = (i + 1) * chunk_length_ms
        
        # Adjust the end time if it exceeds the total duration.
        if end_time > total_duration:
            end_time = total_duration
        
        # Extract the current chunk of audio.
        chunk = audio[start_time:end_time]
        
        # Export the chunk to an MP3 file for transcription.
        chunk.export(str(i)+".mp3", format="mp3")
        
        # Read the exported audio file and perform transcription.
        with open(str(i)+".mp3",'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file, 
                response_format="text",
                language=language_input_value,
                prompt=prompt_input_value,
                temperature=temperature_input_value
            )
            # Concatenate the transcription to the existing string_transcript_audio.
            string_transcript_audio = string_transcript_audio + transcript + ' '
    
    # Display completion message for transcription in the Streamlit app.
    st.write('Transcription Done!','\n')
    
    # Store the final transcription in Transcript_final variable.
    Transcript_final = string_transcript_audio
    
    # Initialize a list to store segmented text.
    t_list = []
    
    # Determine the number of words per segment based on the specified max_len.
    words_per_segment = max_len
    words = Transcript_final.split()
    
    # Segment the entire transcription.
    for i in range(0, len(words), words_per_segment):
        segment = " ".join(words[i:i + words_per_segment])
        t_list.append(segment)
    
    # Return the segmented text and full transcription.
    return t_list, string_transcript_audio

# Function to extract the file ID from a Google Drive link and generate a download link.
def convert_extract_file_id(gdrive_link):
    # Use regular expression to find the file ID from the Google Drive link.
    match = re.search(r"/d/(\S+?)/", gdrive_link)
    if match:
        file_id = match.group(1)
    else:
        pass  # Do nothing if the regex match is not found.

    # Generate a download link using the extracted file ID.
    download_link = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    return download_link

# Function for processing audio files using the Audiogest API.
def audio_processor_audiogest(link_input_value, max_len, string_transcript_audio, language_input_value, prompt_input_value, num_speakers_input_value, wait_time_input_value):
    # Retrieve Audiogest API key from Streamlit secrets.
    audiogest_key = st.secrets["audiogest_key"]

    # Define Audiogest API endpoints and headers.
    transcribe_endpoint = "https://audiogest.app/api/transcripts"
    headers = {
        "Content-type": "application/json",
        "Authorization": f"Bearer {audiogest_key}",
    }

    # Convert Google Drive link to a direct download link.
    link_transcribe = convert_extract_file_id(link_input_value)

    # Create the request body for Audiogest transcription.
    body = {
        "url": link_transcribe,
        "name": "file.mp3", 
        "numSpeakers": num_speakers_input_value,
        "language": language_input_value,
        "vocabulary": prompt_input_value
    }

    try:
        # Make a POST request to initiate the Audiogest transcription process.
        response = requests.post(transcribe_endpoint, headers=headers, data=json.dumps(body))

        # Check if the request was successful (status code 200).
        if response.status_code == 200:
            data = response.json()
            Transcript_ID = data.get("transcriptId", "Not available")
            st.write('Audiogest transcribing process started','\n')
        else:
            st.write('Audiogest error','\n')

    except requests.RequestException as e:
        st.write('Audiogest error','\n')

    # Wait for the specified duration to simulate the transcription process.
    progress_text = f"Transcription in progress. Please wait for {wait_time_input_value} minutes."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep((wait_time_input_value*60)/100)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(1)
    my_bar.empty()

    # Retrieve the transcription results after the waiting period.
    transcriptId = Transcript_ID
    transcript_endpoint = f"https://audiogest.app/api/transcripts/{transcriptId}"

    try:
        # Make a GET request to retrieve the transcription results.
        response = requests.get(transcript_endpoint, headers=headers)

        # Check if the request was successful (status code 200).
        if response.status_code == 200:
            transcript_data = response.json()
            st.write('Transcription process done!','\n')
        else:
            st.write('Audiogest error','\n')

    except requests.RequestException as e:
        st.write('Audiogest error','\n')

    # Process and format the retrieved transcription data.
    for i in range(len(transcript_data['segments'])):
        string_transcript_audio = string_transcript_audio + '<'+ transcript_data['segments'][i]['speaker'] +'>' + ': \n' + transcript_data['segments'][i]['text'] + "\n\n"

    # Store the final transcription in Transcript_final variable.
    Transcript_final = string_transcript_audio
    
    # Initialize a list to store segmented text.
    t_list = []
    
    # Determine the number of words per segment based on the specified max_len.
    words_per_segment = max_len
    words = Transcript_final.split()
    
    # Segment the entire transcription.
    for i in range(0, len(words), words_per_segment):
        segment = " ".join(words[i:i + words_per_segment])
        t_list.append(segment)
    
    # Return the segmented text and full transcription.
    return t_list, string_transcript_audio

# Check if the chosen source type is 'pdf'.
if file_type == 'pdf':
    # Check if a single PDF file is uploaded.
    if uploaded_file is not None and len(uploaded_file) == 1:
        # Process the single PDF file and extract text.
        t_list, full_text = pdf_processor(uploaded_file, max_len)
        
        # Check the selected operation option for note-making.
        if operation_option == "General Note Making":
            # Generate notes using the single PDF file.
            Notes_final_ans = Note_maker(model_option, t_list[0], st.secrets["openai_key"], prompt_option, prompt_area_text, context_file_contents)
        elif operation_option == "Custom Topic Input":
            # Generate notes for each topic using the single PDF file.
            Notes_final_ans = Multi_Custom_Note_maker(uploaded_file, model_option, full_text, st.secrets["openai_key"], topics_input_list, prompt_option, prompt_area_text)

        # Set the desired file name for downloading.
        file_actual_name = file_title + '.txt'
        # Create a download button for the generated call notes.
        st.download_button('Download Call Notes', Notes_final_ans, file_name=file_actual_name, type="primary")
        # Stop the execution to avoid further display of Streamlit app content.
        st.stop()

    # Check if multiple PDF files are uploaded.
    if uploaded_file is not None and len(uploaded_file) > 1:
        # Process the multiple PDF files and extract text.
        t_list, full_text = pdf_processor(uploaded_file, max_len)
        
        # Check the selected operation option for note-making.
        if operation_option == "General Note Making":
            # Generate combined notes using multiple PDF files.
            Notes_final_ans = Multi_Note_maker(uploaded_file, model_option, t_list, st.secrets["openai_key"], prompt_option, prompt_area_text, context_file_contents)
        elif operation_option == "Custom Topic Input":
            # Generate notes for each topic using multiple PDF files.
            Notes_final_ans = Multi_Custom_Note_maker(uploaded_file, model_option, full_text, st.secrets["openai_key"], topics_input_list, prompt_option, prompt_area_text)

        # Set the desired file name for downloading.
        file_actual_name = file_title + '.txt'
        # Create a download button for the generated call notes.
        st.download_button('Download Call Notes', Notes_final_ans, file_name=file_actual_name, type="primary")
        # Stop the execution to avoid further display of Streamlit app content.
        st.stop()
# Initialize an empty string for storing the audio transcript.
string_transcript_audio = ''

# Check if the chosen source type is 'audio'.
if file_type == 'audio':
    # Check if audio file(s) are uploaded.
    if uploaded_file is not None and len(uploaded_file) != 0:
        # Process the audio file using the Whisper ASR model and extract the transcript.
        t_list, string_transcript_audio = audio_processor_whisper(uploaded_file[0], max_len, string_transcript_audio, language_input_value, prompt_input_value, temperature_input_value)

        # Set the desired file name for downloading the audio transcript.
        file_transcript_actual_name = file_title + '_transcript.txt'
        # Create a download button for the audio transcript.
        st.download_button('Download Transcript', string_transcript_audio, file_name=file_transcript_actual_name)

        # Check the selected operation option for note-making.
        if operation_option == "General Note Making":
            # Generate notes using the extracted transcript.
            Notes_final_ans = Note_maker(model_option, t_list, st.secrets["openai_key"], prompt_option, prompt_area_text, context_file_contents)
        elif operation_option == "Custom Topic Input":
            # Generate notes for each topic using the extracted transcript.
            Notes_final_ans = Multi_Custom_Note_maker(uploaded_file, model_option, [string_transcript_audio], st.secrets["openai_key"], topics_input_list, prompt_option, prompt_area_text)

        # Set the desired file name for downloading the call notes.
        file_actual_name = file_title + '.txt'
        # Create a download button for the generated call notes.
        st.download_button('Download Call Notes', Notes_final_ans, file_name=file_actual_name, type="primary")    
        # Stop the execution to avoid further display of Streamlit app content.
        st.stop()

# Check if the chosen source type is 'gdrive link(public access)'.
if file_type == 'gdrive link(public access)':
    # Check if the Google Drive public access link is provided.
    if link_input:
        # Process the audio from the provided Google Drive link using the Audiogest ASR model and extract the transcript.
        t_list, string_transcript_audio = audio_processor_audiogest(link_input_value, max_len, string_transcript_audio, language_input_value, prompt_input_value, num_speakers_input_value, wait_time_input_value)

        # Set the desired file name for downloading the audio transcript.
        file_transcript_actual_name = file_title + '_transcript.txt'
        # Create a download button for the audio transcript.
        st.download_button('Download Transcript', string_transcript_audio, file_name=file_transcript_actual_name)

        # Check the selected operation option for note-making.
        if operation_option == "General Note Making":
            # Generate notes using the extracted transcript.
            Notes_final_ans = Note_maker(model_option, t_list, st.secrets["openai_key"], prompt_option, prompt_area_text, context_file_contents)
        elif operation_option == "Custom Topic Input":
            # Generate notes for each topic using the extracted transcript.
            Notes_final_ans = Multi_Custom_Note_maker(uploaded_file, model_option, [string_transcript_audio], st.secrets["openai_key"], topics_input_list, prompt_option, prompt_area_text)

        # Set the desired file name for downloading the call notes.
        file_actual_name = file_title + '.txt'
        # Create a download button for the generated call notes.
        st.download_button('Download Call Notes', Notes_final_ans, file_name=file_actual_name, type="primary")    
        # Stop the execution to avoid further display of Streamlit app content.
        st.stop()

