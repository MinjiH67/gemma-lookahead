import torch
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "/kaggle/input/gemma-3/transformers/gemma-3-1b-pt/1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def get_lookahead_sequences(model, tokenizer, hypotheses, n_branch_tokens=5, device='cuda'):
    assert len(hypotheses.shape) == 2 and hypotheses.shape[0] == 1, "Expected input shape (1, seq_len)"
    # stores how long the prompt is
    n_tokens_so_far = hypotheses.shape[1]
    hypotheses = hypotheses.to(device)
    past_key_values = DynamicCache() # hold key/value

    with torch.no_grad():
        outputs = model(hypotheses, output_hidden_states=True, past_key_values=past_key_values)

    # Get top-k tokens from last position
    branch_tokens = outputs.logits[0, -1].topk(n_branch_tokens).indices.to(device)
    branched_output_logits = outputs.logits[0, -1]
    print(tokenizer.decode(branch_tokens))
    print("Branch tokens shape:", branch_tokens.shape)  # Expected: (5,)
    assert branch_tokens.shape == (n_branch_tokens,)

    # Repeat past_key_values for each branch
    for i in range(len(past_key_values.key_cache)):
        past_key_values.key_cache[i] = past_key_values.key_cache[i].repeat(n_branch_tokens, 1, 1, 1).to(device)
        past_key_values.value_cache[i] = past_key_values.value_cache[i].repeat(n_branch_tokens, 1, 1, 1).to(device)

    # Fixes the internal tracking 
    past_key_values.reorder_cache(torch.arange(n_branch_tokens, device=device))

    # Start sequences from the branch tokens
    sequences = branch_tokens.unsqueeze(1)
    print("Initial sequences shape:", sequences.shape)  # Expected: (5, 1)
    assert sequences.shape == (n_branch_tokens, 1)

    position_id = n_tokens_so_far
    loop_output_logits = []

    for step in range(2):  # Generate 2 more tokens
        print(f"\n--- Step {step + 1} ---")
        print("Current sequences shape before generation:", sequences.shape)

        cache_position_tensor = torch.tensor([position_id], device=device)  # Convert to tensor 
        #cache_position = torch.full((1,), position_id_for_final_token, dtype=int, device=device)
        # Keep attention mask as is to tell the model to fully attend to each n_branch numbered tokens
        attention_mask = torch.ones((n_branch_tokens,1), dtype=torch.long, device=device)
        #cache_position = torch.full((n_branch_tokens,), position_id, dtype=torch.long, device=device)
        #attention_mask = torch.ones(sequences.shape, dtype=torch.long, device=device)
        print("Before generation:")
        print("past_key_values key shape:", past_key_values.key_cache[0].shape)  # Should start as (5, ..., ..., ...)
        #print("cache_position shape:", cache_position.shape)                     # Should be (5,)
        print("attention_mask shape:", attention_mask.shape)                     # Should be (5, 1) (1,1)


        try:
            with torch.no_grad():
                current_input = sequences[:, -1:]
                print("Input to model (last token):", current_input.shape)  # Expected: (5, 1)
                assert current_input.shape == (n_branch_tokens, 1)

                model_outs = model(
                    current_input,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                    cache_position=cache_position_tensor, #cache_position
                    attention_mask=attention_mask
                )
                print("model_outs type:", type(model_outs))
                print("model_outs logits shape:", model_outs.logits.shape) 
                loop_model_logits = model_outs.logits
                print("model_outs past_key_values shapes:")
                if hasattr(model_outs, "past_key_values"):
                    if isinstance(model_outs.past_key_values, tuple) and len(model_outs.past_key_values) > 0:
                        print("First layer k/v shapes:", 
                              model_outs.past_key_values[0][0].shape, 
                              model_outs.past_key_values[0][1].shape)
        except Exception as e:
            print("Error during model forward pass:", e)
            raise

        next_token_logits = model_outs.logits[:, -1]
        print(next_token_logits)
        print("Next token logits shape:", next_token_logits.shape)  # Expected: (5, vocab_size)
        assert next_token_logits.shape[0] == n_branch_tokens

        next_tokens = next_token_logits.argmax(dim=-1)
        print("Next tokens shape:", next_tokens.shape)  # Expected: (5,)
        assert next_tokens.shape == (n_branch_tokens,)

        sequences = torch.cat([sequences, next_tokens.unsqueeze(1)], dim=1)
        print("Updated sequences shape:", sequences.shape)  # Should grow (5, 2), then (5, 3)

        loop_output_logits.append(loop_model_logits)
        position_id += 1

    print(sequences)
    return sequences, branched_output_logits, loop_output_logits  # Final shape: (5, 3)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_text = "I am"
input_ids = tokenizer(input_text, return_tensors='pt').input_ids
results, branched_token_logit_2, all_logits = get_lookahead_sequences(model, tokenizer, input_ids, device=device)
print(branched_token_logit_2)

# Check if branched logits are equal
are_equal = torch.equal(branched_token_logit_2, branched_logits)
print(are_equal)

loop_logits_list = []
for group in loop_logits:
    # group has shape (5, 1, N), so we squeeze the middle dimension
    squeezed = group.squeeze(1)  # shape becomes (5, N)
    # then split into list of tensors
    loop_logits_list.extend(list(squeezed))

are_equal = (
    len(loop_logits_list) == len(all_logits) and
    all(torch.allclose(a, b, atol=1e-4) for a, b in zip(loop_logits_list, all_logits))
)
print(are_equal)