from langchain_community.1lms.llamacpp import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 
from langchain.embeddings import LlamaCppEmbeddings 
from model. language_model import LanguageModel 

class Llama_LLM(LanguageModel) :

	def init _(self, model_path: str, input_size: int) None: 
		# super()._ init () 
		if input_size > 4096: 
			raise ValueError("Input or output size exceeds 4096 tokens limit for Llama2 models") 
		self.model_path = model_path 
		callback_manager = CallbackManager ( [StreamingStdOutCallbackHandler()]) 
		self.model = LlamaCpp(model_path=model_path, callback_manager=callback _manager, verbose=True, n_ctx=input_size, use_mlock=True) 
	
	
	def generate_text(self, prompt: str, max_tokens: int, temperature: float): 
		# Default method within llama2 model to query 
		# Using it here to reason over the context 
		if max_tokens > 4096: 
			raise ValueError("Input or output size exceeds 4096 tokens limit for Llama2 models") 
		output = self.model(prompt, max_tokens, temperature) 
		return output 
	
	
	def setup_embeddings(self, input_size: int): 
		# Default method within llama2 model to query 
		# Using it here to reason over the context 
		embeddings = LlamaCppEmbeddings (model_path=self.model_path, n_ctx=input_size) 
		return embeddings