from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

def common_concatenated_prefix(list_of_word_lists):
    """
    Given a list of lists (each a sequence of words), return a concatenated prefix
    that is common to all lists while ensuring that it ends at a word boundary for every sequence.
    """
    if not list_of_word_lists:
        return ""

    current_prefixes = ["".join(word_list[:len(word_list)]) for word_list in list_of_word_lists]
    
    # If all concatenated prefixes are the same, continue
    if len(set(current_prefixes)) <= 1  and len(current_prefixes[0]) > 1 :
        return True  # Return the last valid common concatenation

    # If we reach here, the entire shortest sequence is a valid common prefix
    return False


def generate_text_segments(models, tokenizers, context, max_length=150):
    """
    Generate a text segment for each model such that each segment ends at the common 
    word-level boundaries (across all tokenizers).
    
    Args:
        models (dict): Dictionary mapping model names to model objects.
        tokenizers (dict): Dictionary mapping model names to tokenizer objects.
        context (str): The context prompt.
        max_length (int): Maximum number of tokens to generate per model.
    
    Returns:
        dict: Mapping from model name to the generated text segment.
    """

    context_encoded = {
        name: tokenizers[name].encode(context, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        for name in models
    }
    
    generated_tokens = {name: [] for name in models}

    finished_segments = {}

    for _ in range(max_length):
        for name in models:
            if name in finished_segments:
                continue  # Skip models that already have a complete segment.
            
            model_device = next(models[name].parameters()).device
            context_encoded[name] = context_encoded[name].to(model_device)

            # 1. Generate one new token for this model
            new_token = models[name].generate(
                context_encoded[name],
                max_new_tokens= 1,
                do_sample=True,
                attention_mask=torch.ones_like(context_encoded[name]).to(model_device),  # Add attention mask
                pad_token_id=tokenizers[name].pad_token_id or tokenizers[name].eos_token_id,  # Set pad token
                early_stopping=True
            )[:, -1:]
            
            generated_tokens[name].append(new_token)
            context_encoded[name] = torch.cat([context_encoded[name], new_token], dim=-1).to("cpu")
            
            # 2. Decode the generated tokens to text segments.
            candidate_tokens = torch.cat(generated_tokens[name], dim=-1)
            candidate_text = tokenizers[name].decode(candidate_tokens[0])     

            # 3. For each tokenizer, split candidate_text into words.
            tokenized_words = []
            for tok in tokenizers.values():
                # For now, it is just split, to modify
                # words = candidate_text.split()
                # tokenized_words.append(words)

                word_boundaries = []
                if tok.backend_tokenizer.pre_tokenizer:
                    word_list = tok.backend_tokenizer.pre_tokenizer.pre_tokenize_str(candidate_text)
                    for word_id in word_list:
                        if word_id is not None:
                            start, end = word_id[1]
                            word_boundaries.append((start, end))
                    words_sequence = []
                    for start, end in word_boundaries:
                        word = candidate_text[start:end]
                        words_sequence.append(word)
                else:
                    words_sequence = candidate_text.split()
                tokenized_words.append(words_sequence[:-1])
            
            # 4. Compute the longest common prefix (word-level) among all tokenizations without last word
            common_prefix_words = common_concatenated_prefix(tokenized_words)

            # 5. If the common prefix is a complete sentence, store it as a finished segment.
            if common_prefix_words:
                finished_segments[name] = "".join(tokenized_words[0])
                continue
        
        # If all models have generated a complete segment, we can exit early.
        if len(finished_segments) == len(models):
            break
        
    return finished_segments


def compute_perplexity(models, tokenizers, text_segment):
    """
    Compute perplexity for a given text.
    """
    mean_perplexity = 0
    for name in models:
        encodings = tokenizers[name](text_segment, return_tensors="pt")
        input_ids = encodings.input_ids.to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            outputs = models[name](input_ids, labels=input_ids)
            loss = outputs.loss
        perplexity = torch.exp(loss)
        mean_perplexity += perplexity.item()
    return mean_perplexity / len(models)


def rerank_candidates(models, tokenizers, candidates):
    """
    Compute average perplexity from both models and sort candidates.
    Lower average perplexity is considered better.
    """
    ranked = []
    for candidate in candidates:
        mean_ppl = compute_perplexity(models, tokenizers, candidate)
        ranked.append((candidate, mean_ppl))
    ranked.sort(key=lambda x: x[1])  # lower perplexity is better
    return ranked

class CoolFusion:
    def __init__(self, models, tokenizers):
        self.models = models
        self.tokenizers = tokenizers

    def generate_segments(self, context, max_length=150):
        text_segments = generate_text_segments(self.models, self.tokenizers, context, max_length)
        # concatenate the text segments with the context
        text_segments = {name: context + text_segments[name] for name in text_segments}

        rekank = rerank_candidates(self.models, self.tokenizers, list(text_segments.values()))

        # if the best candidate has end of sentence, return it
        if self.tokenizers[list(text_segments.keys())[0]].eos_token in rekank[0][0]:
            return rekank[0][0], False
        
        return rekank[0][0], True


    def generate(self, context, max_length=150):
        # initial generation to the context:
        generated_text = context

        # generate segments until the end of the text, or until the max_length is reached
        while len(generated_text.split()) < max_length:
            segment, end = self.generate_segments(generated_text, max_length)
            # if end of sentence is reached, break
            if not end:
                generated_text = segment
                break

            generated_text = segment

        return generated_text
        

