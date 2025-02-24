import streamlit as st
import asyncio
from groq import AsyncGroq
import uuid

async def generate_response(client, messages, model):
    stream = await client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0.6,
        max_tokens=1024,
        top_p=1,
        stream=True,
    )

    response = ""
    async for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            response += content
            yield content

def get_chat_name(chat):
    for message in chat:
        if message['role'] == 'user':
            return message['content'][:30] + '...' if len(message['content']) > 30 else message['content']
    return "New Chat"

async def main():
    st.title("Chatbot")

    # Disclaimer
    st.warning(
        "Disclaimer: This Groq-powered chatbot may occasionally provide incorrect or inconsistent results. "
        "Please verify any important information and use the responses as a starting point for further conversation."
    )

    # Initialize session state
    if 'chats' not in st.session_state:
        st.session_state.chats = {}
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = None

    # Sidebar
    with st.sidebar:
        # API Key input
        api_key = st.text_input("Enter Groq API Key:", type="password")
        if api_key:
            st.session_state['api_key'] = api_key

        # Model selection
        model = st.selectbox("Select Model", ["deepseek-r1-distill-qwen-32b",
                                              "deepseek-r1-distill-llama-70b-specdec",
                                              "deepseek-r1-distill-llama-70b",
                                              "gemma2-9b-it","llama3-8b-8192",
                                               "llama-3.3-70b-versatile", 
                                               "llama-3.1-8b-instant", "llama-guard-3-8b",
                                               "llama3-70b-8192", "llama3-8b-8192",
                                               "mixtral-8x7b-32768","qwen-2.5-coder-32b",
                                               "qwen-2.5-32b","llama-3.2-11b-vision-preview",
                                               "llama-3.2-90b-vision-preview"])

        # New Chat button
        # if st.button("New Chat"):
        #     new_chat_id = str(uuid.uuid4())
        #     st.session_state.chats[new_chat_id] = [
        #         {"role": "system", "content": "You are a helpful assistant."}
        #     ]
        #     st.session_state.current_chat_id = new_chat_id
        if 'new_chat_clicked' not in st.session_state:
            st.session_state.new_chat_clicked = False

        if st.button("New Chat") or st.session_state.new_chat_clicked:
            if not st.session_state.new_chat_clicked:
                new_chat_id = str(uuid.uuid4())
                st.session_state.chats[new_chat_id] = [
                    {"role": "system", "content": "You are a helpful assistant."}
                ]
                st.session_state.current_chat_id = new_chat_id
                st.session_state.new_chat_clicked = True
            else:
                st.session_state.new_chat_clicked = False


        # Clear All Chats button
        if st.button("Clear All Chats"):
            st.session_state.chats.clear()
            st.session_state.current_chat_id = None
            st.success("All chats have been cleared.")

        # Display previous chats
        st.write("Previous Chats:")
        for chat_id, chat in reversed(list(st.session_state.chats.items())):
            col1, col2 = st.columns([4, 1])
            with col1:
                chat_name = get_chat_name(chat)
                if st.button(chat_name, key=f"chat_{chat_id}"):
                    st.session_state.current_chat_id = chat_id
            with col2:
                if st.button("🗑️", key=f"delete_{chat_id}"):
                    del st.session_state.chats[chat_id]
                    if st.session_state.current_chat_id == chat_id:
                        st.session_state.current_chat_id = None
                    st.rerun()

    # Main chat area
    if st.session_state.current_chat_id:
        current_chat = st.session_state.chats[st.session_state.current_chat_id]

        # Display chat history
        for message in current_chat[1:]:  # Skip the system message
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # User input
        user_input = st.chat_input("You:")

        if user_input:
            # Add user message to chat history
            current_chat.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            # Generate response
            if api_key:
                client = AsyncGroq(api_key=api_key)
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""
                    try:
                        async for content in generate_response(client, current_chat, model):
                            full_response += content
                            response_placeholder.markdown(full_response + "▌")
                        response_placeholder.markdown(full_response)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        return

                # Add assistant's response to chat history
                current_chat.append({"role": "assistant", "content": full_response})
            else:
                st.warning("Please enter your Groq API key in the sidebar to start chatting.")
    else:
        # Create a new chat automatically if there are no chats
        if not st.session_state.chats:
            new_chat_id = str(uuid.uuid4())
            st.session_state.chats[new_chat_id] = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
            st.session_state.current_chat_id = new_chat_id
        else:
            st.info("Select a chat from the sidebar or click 'New Chat' to start a conversation.")

if __name__ == "__main__":
    asyncio.run(main())
