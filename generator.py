from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import re

def generate_text(prompt, max_length=200, temperature=0.7, top_k=40, top_p=0.9):
    try:
        model_path = "gpt2-medium"
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=3,
            no_repeat_ngram_size=3,
            num_beams=5,
            early_stopping=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True
        )

        generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        best_output = max(generated_texts, key=len)

        if best_output.startswith(prompt):
            best_output = best_output[len(prompt):].strip()

        # Clean output with regex to remove unwanted phrases (optional, tweak as needed)
        patterns = [
            r'\b(told in the film|which is next output|free view in itunes)\b.*',
            r'\b(story of how.*?told in the film)\b.*',
            r'\b(which is)\b.*',
            r'\b(new treatment pathways)\b(?!.*(specific|such as))'
        ]
        for pattern in patterns:
            best_output = re.sub(pattern, '', best_output, flags=re.IGNORECASE)

        # Remove trailing incomplete short sentences
        if '.' in best_output:
            sentences = best_output.split('.')
            last_sentence = sentences[-1].strip()
            if len(last_sentence.split()) < 5 and last_sentence:
                sentences = sentences[:-1]
            best_output = '. '.join(s.strip() for s in sentences if s.strip()) + '.'
        else:
            best_output = best_output.strip()

        if not best_output:
            best_output = "No coherent continuation generated."

        return best_output

    except Exception as e:
        return f"Error during text generation: {str(e)}"
