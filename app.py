import streamlit as st
st.set_page_config(page_title="Interactive Text Generator", layout="centered")

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

# --- 1. Setup ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "Alina3234/gemma-lookahead"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


# --- 2. Lookahead token logic ---
def get_lookahead_sequences(model, tokenizer, hypotheses, n_branch_tokens=5, device='cuda'):
    assert len(hypotheses.shape) == 2 and hypotheses.shape[0] == 1, "Expected input shape (1, seq_len)"
    n_tokens_so_far = hypotheses.shape[1]
    hypotheses = hypotheses.to(device)
    past_key_values = DynamicCache()

    with torch.no_grad():
        outputs = model(hypotheses, output_hidden_states=True, past_key_values=past_key_values)

    branch_tokens = outputs.logits[0, -1].topk(n_branch_tokens).indices.to(device)
    assert branch_tokens.shape == (n_branch_tokens,)

    for i in range(len(past_key_values.key_cache)):
        past_key_values.key_cache[i] = past_key_values.key_cache[i].repeat(n_branch_tokens, 1, 1, 1).to(device)
        past_key_values.value_cache[i] = past_key_values.value_cache[i].repeat(n_branch_tokens, 1, 1, 1).to(device)

    past_key_values.reorder_cache(torch.arange(n_branch_tokens, device=device))

    sequences = branch_tokens.unsqueeze(1)
    position_id = n_tokens_so_far
    loop_output_logits = []

    for step in range(2):
        cache_position_tensor = torch.tensor([position_id], device=device)
        attention_mask = torch.ones((n_branch_tokens, 1), dtype=torch.long, device=device)

        with torch.no_grad():
            current_input = sequences[:, -1:]
            model_outs = model(
                current_input,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True,
                cache_position=cache_position_tensor,
                attention_mask=attention_mask
            )

        next_token_logits = model_outs.logits[:, -1]
        next_tokens = next_token_logits.argmax(dim=-1)
        sequences = torch.cat([sequences, next_tokens.unsqueeze(1)], dim=1)
        loop_output_logits.append(model_outs.logits)
        position_id += 1

    return sequences, outputs.logits[0, -1], loop_output_logits


def generate_lookahead_text(model, tokenizer, sequence, n_branch_tokens=5, device='cuda'):
    sequences, _, _ = get_lookahead_sequences(model, tokenizer, sequence, n_branch_tokens, device)
    return tokenizer.batch_decode(sequences, skip_special_tokens=True)


# --- 3. Streamlit UI ---
st.title("✍️ Interactive Lookahead Text Generator")

# Initialize session state for prompt
if "prompt" not in st.session_state:
    st.session_state.prompt = ""

# Editable prompt input
st.session_state.prompt = st.text_area("Your Prompt:", value=st.session_state.prompt, height=100)

if st.button("Generate Completions"):
    if st.session_state.prompt.strip():
        input_ids = tokenizer(st.session_state.prompt, return_tensors="pt").input_ids
        try:
            results = generate_lookahead_text(model, tokenizer, input_ids, device=device)
            st.markdown("### ✨ Top Branching Completions:")
            # for i, res in enumerate(results):
            #     st.write(f"{i + 1}. {res}")
            for i, res in enumerate(results):
                if st.button(f"{i + 1}. {res}", key=f"suggestion_{i}"):
                    # Do something when button is clicked
                    st.session_state.prompt += " " + res
                    st.rerun()
        except Exception as e:
            st.error(f"Error during generation: {e}")
    else:
        st.warning("Please enter some text to begin.")


