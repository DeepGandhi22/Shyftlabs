import streamlit as st
import requests

st.title("RAG Application")

# Document Upload Section
with st.expander("Upload Documents"):
    col1, col2 = st.columns(2)
    
    with col1:
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
        if st.button("Upload PDF"):
            if pdf_file:
                files = {"file": (pdf_file.name, pdf_file.getvalue())}
                response = requests.post("http://localhost:8000/upload/", files=files)
                if response.status_code == 200:
                    st.success(response.json()["status"])
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
    
    with col2:
        url = st.text_input("Enter Website URL")
        if st.button("Upload Website"):
            if url:
                response = requests.post(
                    "http://localhost:8000/upload/",
                    data={"url": url}, 
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                if response.status_code == 200:
                    st.success(response.json()["status"])
                else:
                    st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

question = st.text_input("Enter your question")
search_type = st.selectbox("Search Type", ["semantic", "keyword"])

if st.button("Ask"):
    if question:
        try:
            response = requests.post(
                "http://localhost:8000/query/",
                json={"question": question, "search_type": search_type},
                stream=True
            )
            response.raise_for_status() 
            
            answer_container = st.empty()
            full_answer = ""
            
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    full_answer += chunk.decode("utf-8")
                    answer_container.markdown(full_answer)
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with the server: {str(e)}")