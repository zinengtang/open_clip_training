from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-14B-Instruct-1M"

caption_key = "caption"
# caption_key = "org_caption"
ids = [f"{i:05d}" for i in range(220, 101+200)]
# ids = [f"{i:05d}" for i in range(0, 101)]
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="bfloat16",
    token="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"
).to("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_name)



from tqdm import tqdm
prompt = """
Given an input caption describing an image, generate two variants:
Positive Example: A paraphrased version that preserves the exact meaning using synonyms, grammatical reordering,
or structural changes (e.g., active/passive voice).
Negative Example: A minimal, plausible alteration that subtly contradicts the original meaning. Prioritize composi-
tional changes (e.g., swapped roles, spatial relations, object attributes, or verb actions) while keeping lexical overlap
high. The negative should be visually distinct but textually similar to trick models.
Guidelines: Positive Paraphrase:
Use synonyms (“cube” → “square”), reorder clauses (“X beside Y” → “Y next to X”), or adjust syntax (“holding a
leash” → “gripping a dog’s lead”).
Ensure no key details (objects, relationships, attributes) are altered.
Hard Negative:
Swap Roles/Relations: Invert subject-object relationships (“a man riding a horse” → “a horse beside a man”).
Modify Prepositions/Spatial Logic: Change directional/positional cues (“left of” → “under”).
Alter Attributes: Adjust colors, sizes, or quantities (“three red apples” → “two green apples”).
Reorder Phrases with Identical Words: Use the same words in a different order to invert meaning (“plants surrounding
a lightbulb” → “a lightbulb surrounding some plants”). Wrap positive and negative generation with <positive>...</positive> and 
<negative>...</negative>.
Example: Input: “A chef in a white hat is slicing vegetables on a stainless steel counter while a cat watches from the
windowsill.”
Positive: <positive>“A cook wearing a white cap chops veggies on a shiny metal countertop as a feline observes from the
window ledge.”</positive> (Synonym substitution + rephrasing)
Negative: <negative>“A cat in a white hat is slicing vegetables on a stainless steel counter while a chef watches from the
windowsill.”</negative> (Role swap: “chef” ←−→ “cat” + retained details create a contradictory but plausible scene.)
"""
import pandas as pd
print(ids)
for id in ids:
    # Load input sentences and corresponding keys from the parquet file.
    # Ensure that the order of keys corresponds to the order of captions.
    df = pd.read_parquet(f'/scratch/partial_datasets/datacomp/recap_datacomp_1b_data/{id}.parquet')
    input_sentences = df[caption_key].tolist()
    all_keys = df['key'].tolist()

    # Create the full input texts by applying the chat template for each caption.
    all_texts = []
    for sentence in tqdm(input_sentences, desc="Preparing prompts"):
        messages = [
            {"role": "system", "content": "You are Qwen. You are a helpful assistant.\n" + prompt},
            {"role": "user", "content": sentence}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        all_texts.append(text)

    # Set the batch size and number of sequences to generate per input.
    batch_size = 48
    num_return_sequences = 1

    # This list will accumulate dictionaries containing each key and its generated response.
    results = []

    # Process the inputs in batches.
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Generating responses"):
        batch_texts = all_texts[i: i + batch_size]
        batch_keys = all_keys[i: i + batch_size]

        # Tokenize the batch with padding.
        model_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to(model.device)

        # Generate responses for the batch.
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=256,
            num_return_sequences=num_return_sequences
        )

        # Decode the generated responses.
        decoded_responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Since num_return_sequences = 1, each input corresponds to one response.
        for key, response in zip(batch_keys, decoded_responses):
            results.append({"key": key, "response": response})

        # Convert the accumulated results into a DataFrame.
        df_results = pd.DataFrame(results)

        # Save the DataFrame to a Parquet file.
        output_path = f"../gen_aug_text/{caption_key}_datacomp1b_{id}.parquet"
        df_results.to_parquet(output_path, index=False)
        print(f"Saved responses to {output_path}")

