import pandas as pd
import re

def extract_questions(excel_file_path, sheet_name, column_names):
	excel_data = pd.read_excel(excel_file_path, sheet_name, usecols = column_names, header=1)
	str_list = excel_data.to_string(index=false, header = False).split('\n')

	def clean_section(text):
		text = text.replace('\\n',' ')
		# text = re.sub('\n', ' ', text)
		# remove degits
		# text = re.sub('\d+', '', text)
		# remove non_alphanumerics
		#text = re.sub('[^\w\s]','',text)
		
		# strip ledaing and tailing spaces
		text = text.strip()
		return text
		
	str_list_clean = [clean_section(text) for text in str_list]
	return str_list_clean
	
	
	
if __name__ == "__main__":
	excel_file_path = 'User/downloads/reports.xlsx'
	sheet_name = 'Control questionaire'
	column_names = ['Assessor_guidance']
	extracted_data = extract_questions(excel_file_path, sheet_name, column_names)
	
	print(extracted_data)