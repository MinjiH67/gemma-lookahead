import streamlit as st
import json

# --- 1. Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "user", "content": ""}]
    st.session_state.current_role = "user"

# --- 2. Helper functions ---
def render_messages():
    for i, msg in enumerate(st.session_state.messages[:-1]):
        with st.container():
            st.markdown(f"**{'You' if msg['role'] == 'user' else 'Assistant'}:** {msg['content']}")
            if st.button(f"✏️ Edit", key=f"edit_{i}"):
                rewind_to(i)

    # Display the current editable message
    current_msg = st.session_state.messages[-1]
    label = "You" if current_msg["role"] == "user" else "Assistant"
    st.text_area(f"{label}:", value=current_msg["content"], key="message_input", height=100, on_change=update_current_message)

def update_current_message():
    st.session_state.messages[-1]["content"] = st.session_state.message_input

def send_message():
    update_current_message()
    other_role = "assistant" if st.session_state.current_role == "user" else "user"
    st.session_state.messages.append({"role": other_role, "content": ""})
    st.session_state.current_role = other_role
    st.session_state.message_input = ""

def rewind_to(index):
    st.session_state.messages = st.session_state.messages[:index + 1]
    st.session_state.current_role = st.session_state.messages[-1]["role"]
    st.session_state.message_input = st.session_state.messages[-1]["content"]

def start_new_chat():
    st.session_state.messages = [{"role": "user", "content": ""}]
    st.session_state.current_role = "user"
    st.session_state.message_input = ""

def append_token(token):
    if not st.session_state.allow_multi_word and " " in token:
        token = token.split(" ")[0]
    st.session_state.message_input += token
    st.session_state.messages[-1]["content"] = st.session_state.message_input

# --- 3. UI Rendering ---
st.title("Chat Interface")

render_messages()

col1, col2 = st.columns([1, 1])
with col1:
    st.button("Send", on_click=send_message)
with col2:
    st.button("New Chat", on_click=start_new_chat)

st.checkbox("Allow multi-word predictions", key="allow_multi_word", value=False)

# --- 4. Suggestions (Simulated or connect to backend) ---
# Example simulated suggestion function
def get_suggestions(messages, n_branch_tokens=5):
    # Replace this logic with actual model call if needed
    example_tokens = ["Hi!", "Hello", "How can I help you?", "Tell me more.", "Interesting."]
    return example_tokens, example_tokens

suggestions, display_texts = get_suggestions(st.session_state.messages)

st.markdown("### Suggestions:")
cols = st.columns(len(suggestions))
for i, (sugg, text) in enumerate(zip(suggestions, display_texts)):
    with cols[i]:
        if st.button(text, key=f"suggestion_{i}"):
            append_token(sugg)
