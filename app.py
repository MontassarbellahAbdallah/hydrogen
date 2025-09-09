import streamlit as st
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_conversational_chain():
    prompt_template="""answer the question as detailed as possible from the provided context,
    make sure to provide all the details,if the answer is not in the provided context,
    just say "I am sorry, answer is not avaible in this context" don't provide a wrong answer\n\n
    Context: {context} \n\n
    Quetion: {question} \n\n"""

    model=ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    prompt=PromptTemplate( template=prompt_template,input_variables=["question"])
    chain=load_qa_chain(llm=model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(user_question):
    chain=get_conversational_chain()
    response = chain(
        {"question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("", response["output_text"])

def main():
    st.set_page_config(page_title="Question Answering")
    st.header("ask any information from your docs")
    user_question= st.text_input("enter your question")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()    
