from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def longest_common_prefix(list_of_word_lists):
    """
    Given a list of lists (each a sequence of words), return the longest common prefix as a list of words.
    """
    if not list_of_word_lists:
        return []
    min_length = min(len(words) for words in list_of_word_lists)
    prefix = []
    for i in range(min_length):
        word = list_of_word_lists[0][i]
        if all(words[i] == word for words in list_of_word_lists):
            prefix.append(word)
        else:
            break
    return prefix



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
        name: tokenizers[name].encode(context, return_tensors="pt")
        for name in models
    }
    
    generated_tokens = {name: [] for name in models}

    finished_segments = {}

    for _ in range(max_length):
        for name in models:
            if name in finished_segments:
                continue  # Skip models that already have a complete segment.
            
            # 1. Generate one new token for this model.
            new_token = models[name].generate(
                context_encoded[name],
                max_length=context_encoded[name].shape[-1] + 1,
                do_sample=True,
                attention_mask=torch.ones_like(context_encoded[name]),  # Add attention mask
                pad_token_id=tokenizers[name].pad_token_id or tokenizers[name].eos_token_id  # Set pad token
            )[:, -1:]
            
            generated_tokens[name].append(new_token)
            context_encoded[name] = torch.cat([context_encoded[name], new_token], dim=-1)
            
            # 2. Decode the generated tokens to text segments.
            candidate_tokens = torch.cat(generated_tokens[name], dim=-1)
            candidate_text = tokenizers[name].decode(candidate_tokens[0], skip_special_tokens=True)
            
            # 3. For each tokenizer, split candidate_text into words.
            tokenized_words = []
            for tok in tokenizers.values():
                # For now, it is just split, to modify
                words = candidate_text.split()
                tokenized_words.append(words)
            
            # 4. Compute the longest common prefix (word-level) among all tokenizations without last word
            tokenized_words = [words[:-1] for words in tokenized_words]
            common_prefix_words = longest_common_prefix(tokenized_words)
            common_prefix_text = " ".join(common_prefix_words)
            
            if candidate_text == common_prefix_text or candidate_text.endswith(" "):
                finished_segments[name] = candidate_text.strip()
        
        # If all models have generated a complete segment, we can exit early.
        if len(finished_segments) == len(models):
            break

    # For any model that did not produce a "complete" segment within max_length tokens,
    # fallback by trimming the candidate to the common prefix (dropping the last potentially incomplete word).
    for name in models:
        if name not in finished_segments:
            candidate_tokens = torch.cat(generated_tokens[name], dim=-1)
            candidate_text = tokenizers[name].decode(candidate_tokens[0], skip_special_tokens=True)
            words = candidate_text.split()
            # If at least one word is complete, drop the last token; otherwise, use the candidate as is.
            finished_segments[name] = " ".join(words[:-1]) if len(words) > 1 else candidate_text

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
    def __init__(self, models, tokenizers, max_length=50):
        self.models = models
        self.tokenizers = tokenizers
        self.max_length = max_length

    def generate(self, context):
        text_segments = generate_text_segments(self.models, self.tokenizers, context, self.max_length)
        return rerank_candidates(self.models, self.tokenizers, list(text_segments.values()))

