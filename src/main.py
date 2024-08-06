import os

import streamlit as st
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.messages import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Initialize App requirements
os.makedirs(".cache", exist_ok=True)

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_BASE_URL = os.environ["OPENAI_BASE_URL"]
LLM_MODEL = os.environ["LLM_MODEL"]

# Initialize the state of the app
files = []
retriever = None
st.session_state["messages"] = st.session_state.get(
    "messages",
    [
        ChatMessage(role="assistant", content="무엇을 도와드릴까요?"),
    ],
)

# Show title and description.
st.title("Welcome to AIEEV Chat!")
st.write(
    "Upload a document below and ask a question about it – GPT will answer!  \n"
    f"Current model: {LLM_MODEL}"
)


def new_chat():
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="무엇을 도와드릴까요?")
    ]


with st.sidebar:
    st.logo(
        "https://aieev-public.s3.ap-northeast-2.amazonaws.com/assets/images/logos/%E1%84%8B%E1%85%A1%E1%84%8B%E1%85%B5%E1%84%87%E1%85%B3+%E1%84%85%E1%85%A9%E1%84%80%E1%85%A9%E1%84%83%E1%85%B5%E1%84%8C%E1%85%A1%E1%84%8B%E1%85%B5%E1%86%AB-01-crop.png",
        link="https://aieev.com",
    )
    st.title("AIEEV Chat")

    uploaded_files = st.file_uploader(
        "파일을 선택해주세요 (.txt or .md or .pdf)",
        type=("txt", "md", "pdf"),
        accept_multiple_files=True,
    )

    button = st.button("대화내용 초기화")
    if button:
        new_chat()

    st.divider()

    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=OPENAI_API_KEY,
    )

    # Ask user for their OpenAI API key via `st.text_input`.
    # Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
    # via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
    # openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.", icon="🗝️")

    # "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    # "[Homepage](https://aieev.com)"
    # "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"


def init_retriever():
    docs = []
    for file in files:
        # Step 1: 문서 로드
        # loader = TextLoader("data/appendix-keywords.txt")
        if file.endswith(".txt") or file.endswith(".md"):
            loader = TextLoader(file)
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(file)

        # Step 2: 문서 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 500자로 문서를 분할
            chunk_overlap=50,  # 50자의 중복을 허용
            length_function=len,
        )
        docs += loader.load_and_split(text_splitter)

    # Step 3: 벡터 저장소 생성 & 임베딩(문장을 숫자 표현으로 바꾼다!!) -> 저장
    vectorstore = FAISS.from_documents(
        docs,
        embedding=OpenAIEmbeddings(
            api_key=openai_api_key,  # type: ignore
            base_url=OPENAI_BASE_URL,
            # check_embedding_ctx_length=False,
        ),
    )

    # Step 4: 검색기(retriever) -> 나중에 질문(Query) 에 대한 유사도 검색을 하기 위함
    retriever = vectorstore.as_retriever()
    return retriever


if uploaded_files:
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()

        file_path = f"./.cache/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(bytes_data)

        files.append(file_path)

    retriever = init_retriever()


def get_answer_stream(question: str):
    if retriever:
        # Step 5: 프롬프트 작성, context: 검색기에서 가져온 문장, question: 질문
        template = """당신은 문서에 대한 정보를 바탕으로 답변하는 친절한 Assistant 입니다. 무조건, 주어진 Context 바탕으로 답변해 주세요.
        답변에 대한 출처도 함께 제공해 주세요.
        출처는 파일 이름과 페이지 번호로 표기해 주세요.

        #Context:
        {context}

        #Question:
        {question}
        """

        # Step 6: OpenAI GPT-4 모델을 설정
        model = ChatOpenAI(
            api_key=openai_api_key,  # type: ignore
            base_url=OPENAI_BASE_URL,
            model=LLM_MODEL,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
        )
        prompt = ChatPromptTemplate.from_template(template)

        # Step 7: 질문에 대한 답변을 찾기 위한 체인 생성
        retrieval_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )
        return retrieval_chain.stream(question)
    else:
        template = """너는 일반적인 답변을 하는 Assistant야. 문서를 업로드하면 더 정확한 답변을 해줄 수 있어. 친절하게 답변해줘.
        
        #Question:
        {question}
        """
        model = ChatOpenAI(
            api_key=openai_api_key,  # type: ignore
            base_url=OPENAI_BASE_URL,
            model=LLM_MODEL,
            streaming=True,
            # temperature=PROMPT_LIST[prompt_index]["temperature"],
            callbacks=[StreamingStdOutCallbackHandler()],
        )
        prompt = ChatPromptTemplate.from_template(template)

        # Step 8: 질문&답변
        chain = prompt | model | StrOutputParser()
        return chain.stream({"question": user_input})


# Print messages
for message in st.session_state["messages"]:
    st.chat_message(message.role).write(message.content)


user_input = st.chat_input(
    "질문을 입력해 주세요",
    key="user_input",
    # disabled=(not openai_api_key or not uploaded_files),
)

if user_input:
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))
    st.chat_message("user").write(user_input)

    # Generate an answer using the OpenAI API.
    with st.chat_message("assistant"):
        response_container = st.empty()
        response = ""
        for chunk in get_answer_stream(user_input):
            response += chunk
            response_container.markdown(response)

    st.session_state["messages"].append(ChatMessage(role="assistant", content=response))
