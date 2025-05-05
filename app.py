
import streamlit as st
st.set_page_config(page_title="Interactive Text Generator", layout="centered")

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

torch.classes.__path__ = [] # add this line to manually set it to empty.
## Workaround for the issue with torch.classes.__path__ in transformers library
## Reference: https://discuss.streamlit.io/t/message-error-about-torch/90886/6

# --- 1. Setup ---
@st.cache_resource
def load_model():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "Alina3234/gemma-lookahead"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    return model, tokenizer, device

model, tokenizer, device = load_model()


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


# --- 3. Enhanced generation with caching ---
def generate_initial_lookahead(prompt, n_branch_tokens=5):
    """Generate initial lookahead with full prompt tokenization"""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    return generate_lookahead_text(model, tokenizer, input_ids, n_branch_tokens, device), input_ids


def generate_incremental_lookahead(new_token, n_branch_tokens=5):
    """Generate lookahead based on just the new token"""
    # Tokenize just the new token
    input_ids = tokenizer(new_token, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

    full_prompt = st.session_state.prompt
    full_input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)

    return generate_lookahead_text(model, tokenizer, full_input_ids, n_branch_tokens, device), full_input_ids


# --- 4. Streamlit UI ---
st.title("✍️ Interactive Lookahead Text Generator")
st.markdown("This app shows potential continuations as you write. Select a suggestion to continue your text.")

# Initialize session state variables
if "prompt" not in st.session_state:
    st.session_state.prompt = ""

if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

if "last_token_added" not in st.session_state:
    st.session_state.last_token_added = ""

if "input_ids" not in st.session_state:
    st.session_state.input_ids = None

if "regenerate" not in st.session_state:
    st.session_state.regenerate = False

# Function to handle suggestion selection
def select_suggestion(suggestion):
    # Store the original prompt length before adding the suggestion
    original_length = len(st.session_state.prompt)

    # Add the suggestion to the prompt
    st.session_state.prompt += " " + suggestion

    # Store the newly added token for incremental generation
    st.session_state.last_token_added = suggestion

    # Flag that we need to regenerate
    st.session_state.regenerate = True

    # Clear the current suggestions as they're no longer relevant
    st.session_state.suggestions = []

# Prompt input area
prompt_input = st.text_area(
    "Your text:",
    value=st.session_state.prompt,
    height=150,
    key="prompt_area"
)

# Check if the user has manually edited the prompt
if prompt_input != st.session_state.prompt:
    # Update the prompt and clear any cached state
    st.session_state.prompt = prompt_input
    st.session_state.input_ids = None
    st.session_state.suggestions = []
    st.session_state.last_token_added = ""

# Generate button
if st.button("Generate Completions", type="primary"):
    if st.session_state.prompt.strip():
        with st.spinner("Generating suggestions..."):
            try:
                # Generate initial suggestions based on the full prompt
                st.session_state.suggestions, st.session_state.input_ids = generate_initial_lookahead(
                    st.session_state.prompt
                )
                st.session_state.regenerate = False
            except Exception as e:
                st.error(f"Error during generation: {str(e)}")
    else:
        st.warning("Please enter some text to begin.")

# Display suggestions
if st.session_state.suggestions:
    st.markdown("### ✨ Top Branching Completions:")

    # Create columns for better layout (5 suggestions per row)
    cols = st.columns(5)

    for i, suggestion in enumerate(st.session_state.suggestions):
        col_idx = i % 5
        with cols[col_idx]:
            suggestion_text = suggestion.strip()
            if st.button(f"{suggestion_text}", key=f"sugg_{i}"):
                select_suggestion(suggestion_text)
                st.rerun()

# Auto-regenerate after selecting a suggestion
if st.session_state.regenerate and st.session_state.prompt.strip():
    st.session_state.regenerate = False

    with st.spinner("Generating new suggestions..."):
        try:
            # Generate suggestions based on the incremental token
            st.session_state.suggestions, st.session_state.input_ids = generate_incremental_lookahead(
                st.session_state.last_token_added
            )
            st.rerun()
        except Exception as e:
            st.error(f"Error during generation: {str(e)}")

# Show text preview and stats
if st.session_state.prompt:

    # Display character and word count
    char_count = len(st.session_state.prompt)
    word_count = len(st.session_state.prompt.split())
    st.caption(f"Characters: {char_count} | Words: {word_count}")

# Add a clear button
if st.button("Clear All"):
    st.session_state.prompt = ""
    st.session_state.suggestions = []
    st.session_state.input_ids = None
    st.session_state.last_token_added = ""
    st.rerun()