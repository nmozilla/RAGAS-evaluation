Import sys
Import re
Import os
Import json
Import pandas as pd
Import openai
From langchain_open_ai import AzureOpenAIEmbeddings, AzureChatOpenAI
From langchain_community.vectorstores import FAISS
From langchain_core.promps import ChatPromptTemplate
From langchain_community.document_loaders import PyPDFLoader
From langchain.text_splitter import RecursiveCharacterTextSplitter
From langchain_core.messages import SystemMessage
From langchain.chains import RetrievalQA
From langchain.retrievers.multi_query import MultiQueryRetriever
From ragas.metrics import (
Faithfulness,
Answer_relevency,
Context_relevancy)
From langchain.prompts.chat import (
ChatPromptTemplate,
SystemMessagePromptTemplate,
HumanMessagePromptTemplate)
From datasets import Dataset
From ragas import evaluate

Sys.path.append(".")
From utils.parse_excel import extract_questions

Openai.api_type = "azure"
Openai_api_base = ""
Openai.api_version = "2023-09-15"
AZURE_OPENAI_ENDPOINT = "https://oai.openai.azure.com"
Openai.api_key = os.getenv("OPENAI_API_KEY")

Os.environ["AZURE_OPENAI_ENDPOINT"] = "https://oai.openai.azure.com"
Pdf_path = 'Users/neda/repos/…"
Output_pdf_path = "./docs/summary.pdf"

Excel_file_path = ""
Sheet_name = "questionnaire'
Column_name = ['guidance']
Questions = extract_questions(Excel_file_path, Sheet_name, Column_name)
Questions = Questions[:5]


def row_to_parag(tables, page_correction_val = 5) -> list:
	All_paragraphs = []
	For table in tables:
		If table.parsing_report['accuracy'] >85:
		Df_page_number = table.parsing_report['page']-page_correction_val
		Df = table.df
		Df.columns = df.iloc[0]
		Df = df.drop(0)
		Df['page_number'] = df_page_number
		Headers = list(df.columns)
		Df[df.columns[0]] = df[df.column[0]].replace('',np.nan)
		Df[df.columns[0]] = df[df.column[0]].fillna(method='fill')
		Paragraph = [f"{' '.join([f'{header}:{str(row[header])}' for header in headers])}' for index, row in df.iterrows()]
		All_paragraphs += paragraph
	Clean_paragraphs = [s.re[lace('\n','') for s in all_paragraphs]
	return Clean_paragraphs

def evaluation(question, vectorstore):
	retriever = vectorstore.as_retriever(search_kwargs = {"k":7, "fetch_k":, search_type="mmr")
	# print(help(retriever)
	# template = ("YOu are a TA with 30 years of experience, "
	#"given the context using the information answer the following question."
	#" you can use information in the evidence below to reason over the context. in your answer, provide the criteria number (CC*) and page number as a reference "
	#"like 'Criteria numbers : and 'page Numbers'. Additionally provide the text from the context that supports your answer in the sections 'Text in "
	#"service auditor Test of controls:'. Finally provode your explanation in the 'Explanation' section. If you can't asnwer the question using the required information"
	#" please mention ask my boss."
	# "Context:{context}"
	# "Question:{question}"
	# "Evidence:" + evidence)
	model = AzureChatOpenAI(deployment_name = "gpt35", model_name="gpt35", temprature=0, verbose=True)
	retriever_from_llm = MultiQueryRetriever.from_llm(retriever = retriever, llm = model, parser_key= "lines")
	unique_docs = retriever_from_llm.get_relevant_documents(query=question)
	print("unique_docs:", unique_docs);quit()
	template = ("You are a TA evaluating a report provided by a student."
	" Given the context and using the information answer the question below:"
	"in your answer provide the criteria number (cc) and page number as a reference like 'Criteria number : and Page Number:' Followed by your assessment."
	"If you are unable to answer the question based on the given information, plase state that in your response."
	"Context:{context}\n"
	"Question: {question}\n")
	prompt = ChatPromptTemplate.from_messages([SystemMessagePromptTemplate.from_template(template)])
	qa_chain = RetrievalQA.from_chain_type(model, retriever = retriever, return_source_documents = True, chain_type_kwargs={'prompt':prompt}, chain_type="stuff")
	result = qa_chain({"query": question})
	documents = result['source_documents']
	for i, doc in enumerate(documents):
		text = doc.page_content
		documents[i] = text
	result['source_documents'] = documents
	df = pd.DataFrame(result)
	df['question'] = df['query']
	df['answer'] = df['result']
	df['contexts'] = df['source_documents']
	df['contexts'] = df['contexts'].apply(lambda x: [x])
	data_dict = {'questions': df['question'].to_list(),
	'context':df['contexts'].to_list(),
	'answer':df['answer'].to_list()}
	my_dataset = Dataset.from_dict(data_dict)
	eval_results - evaluate(my_dataset, metrics=[faithfulness, answer_relevancy, context_relevancy], llm = model, embeddings = AzureOpenAIEmbeddings())
	return result, eval_results


def llm_result(questions, vectorstore) -> typing.Tuple[list,list]:
	answers = []
	evals = []
	docs= []
	for index, query in enumerate(questions):
		answer, eval = evaluation(query, vectorstore)
		eval_string = json.dumps(eval)
		answers.append(answer['result'])
		evals.append(eval_string)
		docs.append(answer['source_documents'])
	return answers, evals, docs


def add_to_column(file_path, sheet_name, column, data, evals, docs):
	
	excel_data = pd.read_excel(file_path, sheet_name=sheet_name)
	excel_data[column] = data
	excel_data[column+"eval_metrics"] = evals
	excel_data[column+"context"] = docs
	
	with pd.ExcelWriter(file_path, engine = 'openpyxl', mode='a', if_sheet_exists='overlay') as writer:
		writer.book = load_workbook(file_path)
		writer.sheets.update(dict((ws.title,ws) for ws in writer.book.worksheets))
		excel_data.to_excel(writer, sheet_name = sheet_name, index = False)


tables = camelot.read_pdf(pdf_path, flavor="lattice", pages="all")
Paragraphs = row_to_parag(tables)

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
table_docs = text_splitter.create_documents(paragraphs)
vector_store = FAISS.from_documents(table_docs, AzureOpenAIEmbeddings())

loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()
pages = pages[:15]
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 15)
docs = text_splitter.split_documents(pages)
vector_store.add_documents(docs)

answers, evals, docs = llm_result(questions, vector_store)

sheet_to_write = 'Questionnaire'
column_to_write = 'LLM Observation'
excel_file_path = '/user/nd/downloads/llm_obs.xlsx'
add_to_column(excel_file_path, sheet_to_write, column_to_write, answers, evals, docs)

 