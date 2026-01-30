from typing import List
from transformers import PreTrainedTokenizer

B_INST, E_INST = "[INST]", "[/INST]"

def tokenize_llama_chat(
    tokenizer: PreTrainedTokenizer,
    user_input: str,
    model_output: str = None,
) -> List[int]:
    input_content = ""
    input_content += f"{B_INST} {user_input.strip()} {E_INST}"
    if model_output is not None:
        input_content += f" {model_output.strip()}"
    return tokenizer.encode(input_content)