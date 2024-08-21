 import sys 
 sys. path. append (".") 
 import os 
 import re 
 import openai
 from PyPDF2 import PdfReader
 from utils-parse_excel import extract_excel_columns 
 from openai import OpenAI 
 
 openai.api_type = "azure" 
 openai.api_base = "https://oai.openai.azure.com/" 
 # openai.api_base = "https://text-embedding-ada-002.openai.azure.com/* 
 openai.api_version = "2023-09-15-preview" 
 openai-api_key = os-getenv("OPENAI_API_KEY")
 
 os. environ ["AZURE_OPENAI_ENDPOINT"] = "https://text-embedding-ada-002.openai.azure.com/" 
 client = OpenAI(api_key=os -getenv("OPENAI_API_KEY") ) 

def extract_text_from_pdf(pdf_path: None|str, range_pages: None|tuple) -> str: 
	with open(pdf_path, "rb") as file: 
		pdf_reader = PdfReader (file) 
		print (pdf_reader.metadata) 
		text = ""
		start, end = (0, len(pdf_reader-pages)) if range_pages == None else range_pages [0], range_pages [1]
		print(start, end) 
		for page_num in range(start, end): 
			page = pdf_reader. pages [page_num] 
			text + page.extract_text() 
	return text 

def summarise(pdf_text:str, question:str, temperature:float, max_tokens: int) → str: 
	PROMPT = f"You are security assessor. Below is an extract from the security report of a company.
				{question} Provide citations or references in the text for these. \ 
				\n\n# Start of Report\n{pdf_text}\n# End of Report" 
	response = client.chat.completions.create( 
											  model="gpt35", 
											  messages=[{ 
											  "role": "user", 
												"content": PROMPT, 
												} 
											], 
											temperature=temperature, 
											max_tokens=max_tokens, 
											top_p=1, 
											frequency_penalty=0, 
											presence_penalty=0, 
											stop=None)
	return response ["choices"] [0] ["text"] 


excel_file_path = '/Users/nd/Downloads/Review_empty.xlsx'
# Example usage: 
pdf_path = './docs/proves.pdf' 
output_pdf_path = '•/docs/summary.pdf' 
pdf_text = extract_text_from_pdf(pdf_path, range_pages=(2,10)) # Mention starting page number and ending page number 

sheet name = 'Questionnaire' 
column_names = ['Guidance'] 
all_questions = extract_excel_columns(excel_file_path, sheet_name, column_names)
str_list = all_questions. to_string(index=False, header=False) split('\n") 
str_list = [s.strip() for s in str_list] 
questions = str_list[:2] 


def bot(questions: list): 
	answers = (} 
	for question in questions: 
		response = summarise(pdf_text, question, temperature=0.5, max_tokens=150) 
		answers [question] = response 
	return answers 

print (bot (questions) )