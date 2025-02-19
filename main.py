import os
from prompt import system
from uuid import uuid4 as uid
import streamlit as st
from streamlit import session_state
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()



client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if "session_id" not in session_state:
    session_state["session_id"] = str(uid())

def stream_chat_response(data: str):
    out = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": data}
        ],
        temperature=0.2,
        stream=True
    )

    for chunked in out:
        if chunked.choices and chunked.choices[0].delta.content:
            yield chunked.choices[0].delta.content


st.title("ML Agent")
csv_file = st.file_uploader("Upload a CSV dataset", type="csv")

if csv_file and st.button("Generate"):

    df = pd.read_csv(csv_file)
    csv_text = df.to_csv(index=False)
    with st.chat_message("assistant"):
        response = st.write_stream(stream_chat_response(csv_text))

    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
# if prompt := st.chat_input("How can i assist you further ?"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#
#     with st.chat_message("user"):
#         st.markdown(prompt)
#
#     with st.chat_message("assistant"):
#         pre = "The training process has been completed, and the optimized model is now available. Please proceed with the following tasks:"
#         response = st.write_stream(stream_chat_response(pre + prompt))
#
#     st.session_state.messages.append({"role": "assistant", "content": response})



