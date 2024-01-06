import streamlit as st
import PyPDF2
from io import BytesIO
import openai
import pydub
import requests
import json
import time
import re

st.set_page_config(
    page_title="NoteMaker",
    page_icon=":bookmark_tabs:",
)

st.sidebar.markdown("# Useful Prompts")

st.sidebar.markdown("### General Note Making")
st.sidebar.markdown("- Generate detailed call notes of the conversation for an investment firm.\nGenerate notes pointwise with full sentences under each of all the Important Sections, (Convert all text numbers to numbers, Include all important information and numbers.)\n")

st.sidebar.markdown("### Custom Topic Input")
st.sidebar.markdown("- Generate detailed call notes of the conversation for an investment firm only under this topic-{'<topic>'}. Do not create any subtopics under the topics. Generate notes pointwise under only that topic, (Convert all text numbers to numbers)")

st.sidebar.markdown("### Misc")
st.sidebar.markdown("- Please revise the following text by eliminating any repetitive information and clubbing similar topics under common headings. Ensure that each heading is followed by its relevant points. Do not rewrite any point or make them smaller. Avoid using bold formatting and numbering for the headings.")

st.markdown('# Hello User!')

file_type = st.selectbox(
    'Choose source type [pdf, audio, audiogest-link]:',
    ('pdf', 'audio', 'gdrive link(public access)'))

link_input = st.text_input('Enter the google drive public access link of the file here(if you have chosen "gdrive link(public access)" in the previous question):')
link_input_value = ''
if link_input:
	link_input_value = link_input

with st.expander('Whisper/Audiogest Settings', expanded=False):
	language_input = st.text_input('Enter the language of the audio in "ISO-639-1" format {english = en, hindi = hi}:', value='en')
	prompt_input = st.text_area('Enter your custom prompt which may contain factual words present in the audio:')
	temperature_input = st.number_input("Enter a number between [0,1] for temperature value:", value=0.3)
	num_speakers_input = st.number_input("Enter the number of speakers including the interviewer(for audiogest):", value=2)
	wait_time_input = st.number_input("Enter the time(in minutes) to wait for audiogest transcription:", value=10)	

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

with st.expander('GPT parameters', expanded=False):
	temperature_input_GPT = st.number_input("Enter a number between [0,2] for temperature value of GPT Note Making:", value=1.5)
	top_p_input_GPT = st.number_input("Enter the 'Top p' input value of GPT Note Making:", value=0.6)
	frequency_penalty_GPT = st.number_input("Enter the 'frequency_penalty' input value of GPT Note Making:", value=0.1)
	presence_penalty_GPT = st.number_input("Enter the 'presence_penalty' input value of GPT Note Making:", value=0.1)
	context_file = st.file_uploader("[optional] Context PDF file:", type=["pdf"], accept_multiple_files=False)
	model_option = st.selectbox(
    'Which model would you like to use?',
    ('gpt-4-1106-preview', 'gpt-4', 'gpt-3.5-turbo-1106'))

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

max_len_str = st.text_input('Chunk size :red[[required]] :')
if max_len_str:
	max_len = int(max_len_str)

file_title = st.text_input('File title:')
if file_title:
	pass
else:
	file_title = 'place_holder_for_file_name'

if context_file:
	context_file_contents = ''
	
	reader = PyPDF2.PdfReader(context_file)
			
	for j in range(len(reader.pages)):    
		p = reader.pages[j]
		t = p.extract_text()
			
		context_file_contents = context_file_contents + "\n" + t

else:
	context_file_contents = ''

operation_option = st.selectbox(
    'Which operation do you want to perform?',
    ('General Note Making', 'Custom Topic Input'))

topic_input_file = st.file_uploader("Choose a :red[PDF file containing the topics] arranged properly:", type=["pdf"], accept_multiple_files=False)

user_prompt_input = st.text_input('Enter the comma seperated topics in 1 line (If you have chosen "custom topic input"):')

if topic_input_file:
	input_text = ''
	
	reader = PyPDF2.PdfReader(topic_input_file)
			
	for j in range(len(reader.pages)):    
		p = reader.pages[j]
		t = p.extract_text()
			
		input_text = input_text + "\n" + t

	topics_input_list = [i.strip().replace('\n','') for i in input_text.split('\n \n')]

elif user_prompt_input:
	topics_input_list = [i.strip() for i in user_prompt_input.split(',')]


prompt_area_default_text="Generate detailed call notes of the conversation for an investment firm.\nGenerate notes pointwise with full sentences under each of all the Important Sections, (Convert all text numbers to numbers, Include all important information and numbers.)\n"
prompt_area_text=''
if operation_option:
	if operation_option=='General Note Making':
		prompt_area_default_text = "Generate detailed call notes of the conversation for an investment firm.\nGenerate notes pointwise with full sentences under each of all the Important Sections, (Convert all text numbers to numbers, Include all important information and numbers.)\n"
	elif operation_option=='Custom Topic Input':
		prompt_area_default_text = "Generate detailed call notes of the conversation for an investment firm only under this topic-{'<topic>'}. Do not create any subtopics under the topics. Generate notes pointwise under only that topic, (Convert all text numbers to numbers)"

prompt_area_text = st.text_area("This is the default prompt. If you modify the prompt, please choose 'Use customized prompt' in the next question_____ :red[DO NOT REMOVE THE <topic> PART OF THE PROMPT IN Custom Topic Input]", value=prompt_area_default_text)

prompt_option = st.selectbox(
    ':red[Do you want to use the customized prompt ?]',
    ('Use default prompt', 'Use customized prompt'))

uploaded_file = st.file_uploader("Choose a PDF/Audio file:", type=["pdf","mp3","mp4","m4a","wav"], accept_multiple_files=True)



def Note_maker(model_option, t_list, api_key, prompt_option, prompt_area_text, context_file_contents):
	client = openai.OpenAI(api_key=api_key)
	st.write('[Note Making] Progress update:','\n')
	st.write(1,'/',len(t_list),'\n')
	
	if prompt_option=="Use default prompt":
		actual_prompt = "Generate detailed call notes of the conversation for an investment firm.\nGenerate notes pointwise with full sentences under each of all the Important Sections, (Convert all text numbers to numbers, Include all important information and numbers.)\n"
	elif prompt_option=="Use customized prompt":
		actual_prompt = prompt_area_text
	
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
	
	response = client.chat.completions.create(
	  model=model_option,
	  messages=message_list,
	  temperature=temperature_input_GPT_value,
	  max_tokens=4096,
	  top_p=top_p_input_GPT_value,
	  frequency_penalty=frequency_penalty_GPT_value,
	  presence_penalty=presence_penalty_GPT_value
	)
	
	Notes = []
	
	for i in range(1,len(t_list)):
		
		st.write(i+1,'/',len(t_list),'\n')
		
		Notes.append(response.choices[0].message.content)

		if (model_option=='gpt-3.5-turbo-1106') or (model_option=='gpt-4'):
			try:
				del message_list[2:4]
			except:
				pass
		elif (model_option=='gpt-4-1106-preview'):
			try:
				if len(message_list)>=8:
					del message_list[2:4]
			except:
				pass
		
		message_list.append({
		      "role": "assistant",
		      "content": context_file_contents + response.choices[0].message.content
		    })
	    
		stock = "This is continuation of the transcript.\nGenerate call notes pointwise for an investment firm under each of all the Important Sections, (Convert all text numbers to numbers, Include all important information and numbers.)\n"
		cont = stock + t_list[i]
	    
		message_list.append({
		      "role": "user",
		      "content": cont
		    })
	    
		response = client.chat.completions.create(
		      model=model_option,
		      messages=message_list,
		      temperature=temperature_input_GPT_value,
		      max_tokens=4096,
		      top_p=top_p_input_GPT_value,
		      frequency_penalty=frequency_penalty_GPT_value,
		      presence_penalty=presence_penalty_GPT_value
		    )
	    
	Notes.append(response.choices[0].message.content)

	Notes_Final = ''

	for i in Notes:
		Notes_Final = Notes_Final + i + '\n\n'

	st.write('Process done!','\n')
	return Notes_Final

def Custom_Note_maker(model_option, t_list, api_key, user_prompt_input, prompt_option, prompt_area_text):
	client = openai.OpenAI(api_key=api_key)
	st.write('[Custom Input Note Making] Progress update:','\n')
	Notes = []
	
	for i in range(len(t_list)):
		st.write(i+1,'/',len(t_list),'\n')

		if prompt_option=="Use default prompt":
			actual_prompt = "Generate detailed call notes of the conversation for an investment firm only under these topics-{"+user_prompt_input+"}.\nGenerate notes pointwise under only those topics, (Convert all text numbers to numbers)"
		elif prompt_option=="Use customized prompt":
			actual_prompt = prompt_area_text
		
		message_list = [
		{
		  "role": "system",
		  "content": actual_prompt
		},
		{
		  "role": "user",
		  "content": 'Notes: \n'+t_list[i]
		}
		]
		
		response = client.chat.completions.create(
		model='gpt-4-1106-preview',
		messages=message_list,
		temperature=temperature_input_GPT_value,
		max_tokens=4096,
		top_p=top_p_input_GPT_value,
		frequency_penalty=frequency_penalty_GPT_value,
		presence_penalty=presence_penalty_GPT_value
		)
		
		Notes.append(response.choices[0].message.content)

	topics = [i.strip() for i in user_prompt_input.split(',')]
	
	topics_names = [s.replace(' ', '_') for s in topics]
	
	for j in range(len(topics)):
		globals()[topics_names[j]+"_items"] = ''
	
	for i in range(len(Notes)):
		ex = [i.strip() for i in Notes[i].strip().split('\n\n')]
		if len(ex)>=len(topics):
			for j in range(len(topics)):
				globals()[topics_names[j]+"_items"] = globals()[topics_names[j]+"_items"] + ex[j]
		    
	Custom_notes = ''
	
	for j in range(len(topics)):
		Custom_notes = Custom_notes + globals()[topics_names[j]+"_items"] + '\n\n'

	return Custom_notes

def Multi_Note_maker(uploaded_file, model_option, t_list, api_key, prompt_option, prompt_area_text, context_file_contents):
	client = openai.OpenAI(api_key=api_key)
	Notes_Final_Final = ''
	for j in range(len(t_list)):
		st.write(f'[Note Making {j+1}] Progress update:','\n')
		st.write(1,'/',len(t_list[j]),'\n')
		
		if prompt_option=="Use default prompt":
			actual_prompt = "Generate detailed call notes of the conversation for an investment firm.\nGenerate notes pointwise with full sentences under each of all the Important Sections, (Convert all text numbers to numbers, Include all important information and numbers.)\n"
		elif prompt_option=="Use customized prompt":
			actual_prompt = prompt_area_text
		
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
		
		response = client.chat.completions.create(
		model=model_option,
		messages=message_list,
		temperature=temperature_input_GPT_value,
		max_tokens=4096,
		top_p=top_p_input_GPT_value,
		frequency_penalty=frequency_penalty_GPT_value,
		presence_penalty=presence_penalty_GPT_value
		)
		
		Notes = []
		
		for i in range(1,len(t_list[j])):
			
			st.write(i+1,'/',len(t_list[j]),'\n')
			
			Notes.append(response.choices[0].message.content)

			if (model_option=='gpt-3.5-turbo-1106') or (model_option=='gpt-4'):
				try:
					del message_list[2:4]
				except:
					pass
			elif (model_option=='gpt-4-1106-preview'):
				try:
					if len(message_list)>=8:
						del message_list[2:4]
				except:
					pass
			
			message_list.append({
				"role": "assistant",
				"content": context_file_contents + response.choices[0].message.content
				})
			
			stock = "This is continuation of the transcript.\nGenerate call notes pointwise for an investment firm under each of all the Important Sections, (Convert all text numbers to numbers, Include all important information and numbers.)\n"
			cont = stock + t_list[j][i]
			
			message_list.append({
				"role": "user",
				"content": cont
				})
			
			response = client.chat.completions.create(
				model=model_option,
				messages=message_list,
				temperature=temperature_input_GPT_value,
				max_tokens=4096,
				top_p=top_p_input_GPT_value,
				frequency_penalty=frequency_penalty_GPT_value,
				presence_penalty=presence_penalty_GPT_value
				)
			
		Notes.append(response.choices[0].message.content)

		Notes_Final = ''

		for i in Notes:
			Notes_Final = Notes_Final + i + '\n\n'

		Notes_Final_Final = Notes_Final_Final + uploaded_file[j].name + ' :\n' + Notes_Final + '\n\n\n'
		
	st.write('Process done!','\n')
	return Notes_Final_Final

def Multi_Custom_Note_maker(uploaded_file, model_option, full_text, api_key, topics_input_list, prompt_option, prompt_area_text):
	client = openai.OpenAI(api_key=api_key)
	Notes_Final_Final = '' 
	
	topics = topics_input_list

	if prompt_option=="Use default prompt":
		actual_prompt = "Generate detailed call notes of the conversation for an investment firm only under this topic-{'<topic>'}. Do not create any subtopics under the topics. Generate notes pointwise under only that topic, (Convert all text numbers to numbers)"
	elif prompt_option=="Use customized prompt":
		actual_prompt = prompt_area_text

	for i in range(len(topics)):
		st.write(f"[Custom Input Note Making {i+1}/{len(topics)}] Progress update:",'\n')
		Notes_Final_Final = Notes_Final_Final + "     " + topics[i] + ':\n'

		for j in range(len(full_text)):   
			st.write(j+1,'/',len(full_text),'\n')
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

			response = client.chat.completions.create(
			model='gpt-4-1106-preview',
			messages=message_list,
			temperature=temperature_input_GPT_value,
			max_tokens=4096,
			top_p=top_p_input_GPT_value,
			frequency_penalty=frequency_penalty_GPT_value,
			presence_penalty=presence_penalty_GPT_value
			)
			
			Notes_Final_Final = Notes_Final_Final + "         " + uploaded_file[j].name + ":\n"
			Notes_Final_Final = Notes_Final_Final + response.choices[0].message.content + "\n\n"
		Notes_Final_Final = Notes_Final_Final + "\n\n\n\n"

	return Notes_Final_Final

def pdf_processor(uploaded_file, max_len):
	full_text = ["" for k in range(len(uploaded_file))]
	
	for i in range(len(uploaded_file)):
		reader = PyPDF2.PdfReader(uploaded_file[i])
				
		for j in range(len(reader.pages)):    
			p = reader.pages[j]
			t = p.extract_text()
				
			full_text[i] = full_text[i] + "\n" + t
	
	Transcript_final = full_text
	
	t_list = [[] for k in range(len(Transcript_final))]
	
	words_per_segment = max_len
	for j in range(len(Transcript_final)):
		words = Transcript_final[j].split()
		
		for i in range(0, len(words), words_per_segment):
			segment = " ".join(words[i:i + words_per_segment])
			t_list[j].append(segment)

	return t_list, full_text

def audio_processor_whisper(uploaded_file, max_len, string_transcript_audio, language_input_value, prompt_input_value, temperature_input_value):
	audio = pydub.AudioSegment.from_file(uploaded_file)
	total_duration = len(audio)
	chunk_length_ms = 60000
	num_chunks = total_duration // chunk_length_ms
	
	client = openai.OpenAI(api_key=st.secrets["openai_key"])
	st.write('[Transcription] Progress update:','\n')
	for i in range(num_chunks):
		st.write(i+1,'/',num_chunks,'\n')
		start_time = i * chunk_length_ms
		end_time = (i + 1) * chunk_length_ms
	
		if end_time > total_duration:
			end_time = total_duration
	
		chunk = audio[start_time:end_time]
		chunk.export(str(i)+".mp3", format="mp3")
	
		with open(str(i)+".mp3",'rb') as audio_file:
			transcript = client.audio.transcriptions.create(
					  model="whisper-1", 
					  file=audio_file, 
					  response_format="text",
					  language=language_input_value,
					  prompt=prompt_input_value,
					  temperature=temperature_input_value
					)
			string_transcript_audio = string_transcript_audio + transcript + ' '
	st.write('Transcription Done!','\n')
	Transcript_final = string_transcript_audio
	
	t_list = []
	
	words_per_segment = max_len
	words = Transcript_final.split()
	
	for i in range(0, len(words), words_per_segment):
		segment = " ".join(words[i:i + words_per_segment])
		t_list.append(segment)

	return t_list, string_transcript_audio

def convert_extract_file_id(gdrive_link):
    match = re.search(r"/d/(\S+?)/", gdrive_link)
    if match:
        file_id = match.group(1)
    else:
        pass

    download_link = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    return download_link

def audio_processor_audiogest(link_input_value, max_len, string_transcript_audio, language_input_value, prompt_input_value, num_speakers_input_value, wait_time_input_value):
	audiogest_key = st.secrets["audiogest_key"]

	transcribe_endpoint = "https://audiogest.app/api/transcripts"

	headers = {
		"Content-type": "application/json",
		"Authorization": f"Bearer {audiogest_key}",
		
	}

	link_transcribe = convert_extract_file_id(link_input_value)

	body = {
		"url": link_transcribe,
		"name": "file.mp3", 
		"numSpeakers": num_speakers_input_value,
		"language": language_input_value,
		"vocabulary": prompt_input_value
	}


	try:
		response = requests.post(transcribe_endpoint, headers=headers, data=json.dumps(body))

		if response.status_code == 200:
			data = response.json()
			Transcript_ID = data.get("transcriptId", "Not available")
			st.write('Audiogest transcribing process started','\n')
		else:
			st.write('Audiogest error','\n')

	except requests.RequestException as e:
		st.write('Audiogest error','\n')

	#wait time
	progress_text = f"Transcription in progress. Please wait for {wait_time_input_value} minutes."
	my_bar = st.progress(0, text=progress_text)

	for percent_complete in range(100):
		time.sleep((wait_time_input_value*60)/100)
		my_bar.progress(percent_complete + 1, text=progress_text)
	time.sleep(1)
	my_bar.empty()

	#retrieval
	transcriptId = Transcript_ID

	transcript_endpoint = f"https://audiogest.app/api/transcripts/{transcriptId}"

	headers = {
		"Content-type": "application/json",
		"Authorization": f"Bearer {audiogest_key}",
	}

	try:
		response = requests.get(transcript_endpoint, headers=headers)

		if response.status_code == 200:
			transcript_data = response.json()
			st.write('Transcription process done!','\n')
		else:
			st.write('Audiogest error','\n')

	except requests.RequestException as e:
		st.write('Audiogest error','\n')

	for i in range(len(transcript_data['segments'])):
		string_transcript_audio = string_transcript_audio + '<'+ transcript_data['segments'][i]['speaker'] +'>' + ': \n' + transcript_data['segments'][i]['text'] + "\n\n"

	Transcript_final = string_transcript_audio
	
	t_list = []
	
	words_per_segment = max_len
	words = Transcript_final.split()
	
	for i in range(0, len(words), words_per_segment):
		segment = " ".join(words[i:i + words_per_segment])
		t_list.append(segment)

	return t_list, string_transcript_audio

if file_type == 'pdf':
	if uploaded_file is not None and len(uploaded_file)==1:

		t_list, full_text = pdf_processor(uploaded_file, max_len)
		
		if operation_option == "General Note Making":
			Notes_final_ans = Note_maker(model_option, t_list[0], st.secrets["openai_key"], prompt_option, prompt_area_text, context_file_contents)
		elif operation_option == "Custom Topic Input":
			Notes_final_ans = Multi_Custom_Note_maker(uploaded_file, model_option, full_text, st.secrets["openai_key"], topics_input_list, prompt_option, prompt_area_text)

		file_actual_name = file_title + '.txt'
		st.download_button('Download Call Notes', Notes_final_ans, file_name=file_actual_name, type="primary")
		st.stop()

	if uploaded_file is not None and len(uploaded_file)>1:

		t_list, full_text = pdf_processor(uploaded_file, max_len)
		
		if operation_option == "General Note Making":
			Notes_final_ans = Multi_Note_maker(uploaded_file, model_option, t_list, st.secrets["openai_key"], prompt_option, prompt_area_text, context_file_contents)
		elif operation_option == "Custom Topic Input":
			Notes_final_ans = Multi_Custom_Note_maker(uploaded_file, model_option, full_text, st.secrets["openai_key"], topics_input_list, prompt_option, prompt_area_text)

		file_actual_name = file_title + '.txt'
		st.download_button('Download Call Notes', Notes_final_ans, file_name=file_actual_name, type="primary")
		st.stop()

string_transcript_audio=''

if file_type == 'audio':
	if uploaded_file is not None and len(uploaded_file)!=0:

		t_list, string_transcript_audio = audio_processor_whisper(uploaded_file[0], max_len, string_transcript_audio, language_input_value, prompt_input_value, temperature_input_value)

		file_transcript_actual_name = file_title + '_transcript.txt'
		st.download_button('Download Transcript', string_transcript_audio, file_name=file_transcript_actual_name)

		if operation_option == "General Note Making":
			Notes_final_ans = Note_maker(model_option, t_list, st.secrets["openai_key"], prompt_option, prompt_area_text, context_file_contents)
		elif operation_option == "Custom Topic Input":
			Notes_final_ans = Multi_Custom_Note_maker(uploaded_file, model_option, [string_transcript_audio], st.secrets["openai_key"], topics_input_list, prompt_option, prompt_area_text)

		file_actual_name = file_title + '.txt'
		st.download_button('Download Call Notes', Notes_final_ans, file_name=file_actual_name, type="primary")	
		st.stop()

if file_type == 'gdrive link(public access)':
	if link_input:
		t_list, string_transcript_audio = audio_processor_audiogest(link_input_value, max_len, string_transcript_audio, language_input_value, prompt_input_value, num_speakers_input_value, wait_time_input_value)

		file_transcript_actual_name = file_title + '_transcript.txt'
		st.download_button('Download Transcript', string_transcript_audio, file_name=file_transcript_actual_name)

		if operation_option == "General Note Making":
			Notes_final_ans = Note_maker(model_option, t_list, st.secrets["openai_key"], prompt_option, prompt_area_text, context_file_contents)
		elif operation_option == "Custom Topic Input":
			Notes_final_ans = Multi_Custom_Note_maker(uploaded_file, model_option, [string_transcript_audio], st.secrets["openai_key"], topics_input_list, prompt_option, prompt_area_text)

		file_actual_name = file_title + '.txt'
		st.download_button('Download Call Notes', Notes_final_ans, file_name=file_actual_name, type="primary")	
		st.stop()
