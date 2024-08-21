import numpy as np

def row_to_paragraph(tables, page_correction_val=6)-> list:
	all_paragraphs = []
	all_metadata = []
	
	
	for table in tables:
		if table.parsing_report['accuracy']>90:
			# extract page number from table
			df_page_number = table.parsing_report['page']-page_correction_val
			df = table.df
			df.columns = df.iloc[0]
			df = df.drop(0)
			df["Page_number"] = df_page_number
			
			#replace empty cells with previous non-empty cells
			df[df.columns[0]] = df[df.columns[0]].replace('', np.nan).ffill()
			df_new = df.iloc[:, 0 ].to_frame() # copy the first column to a new DataFrame
			df = df.drop(df.columns[0],axis = 1)
			headers = list(df.columns)
			
			# Convert each row of the dataframe into a sentence
			paragraph = [f"{' '.join([f'{header}: {str(row[header])}' for header in headers])}"
							for ids, row in df.iterrows()]
			
			all_metadata += [row.to_dict() for idx, row in df_new.iterrows()]
			
			all_paragraphs += paragraph
			
	clean_paragraphs = [s.replace('\n','') for s in all_paragraphs]
	return clean_paragraphs, all_metadata
	