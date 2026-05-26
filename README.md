# Abstractive Summary

## Phân chia kiến trúc:

1. Pre-Process Pipeline: Vocabulary and Tokenizer

    - Input:    Corpura (raw text - paragraphs)
    - Output:   
        - Tokenized corpora (list of sentences, each sentence is a list of tokens)
        - Vocabulary (token to ID mapping, ID to token list, token frequency)
    - Components:
        - Tokenizer
        - Vocabulary
        - ~VocabConfig

2. Word Embeddings

    - Input:    
        - Corpura (raw text - paragraphs)
        - Vocabulary (token to id mapping)
    - Output:   
        - Word Embeddings (token to vector mapping, ID to vector mapping)
    - Components:
        - WordEmbedding
        
3. Network Core

    - Input:    
        - Word Embeddings (token to vector mapping)
        - Paragraph (convert to matrix of Vector Embeddings)
    - Workflow:
        - Mapping paragraph (as list of indices) to matrix (list of Embedding Vector)
        - Forward matrix embedding through `Network` to become output hidden state.
        - Projecting output state to expected output (text)
    - Output:   
        - Summary
    - Components:
        - Network



## Hướng dẫn cài đặt (nếu chạy local)

Yêu cầu cài đặt các thư viện trong file: [requirements.txt](./requirements.txt)

Khuyên dùng: venv (Virtual Enviroment)
Hệ điều hành: Windows

```bash
py -m venv .venv                   // [1] Khởi tạo venv cho Windows

.venv\Scripts\activate             // [2] Kích hoạt venv cho Windows

pip install -r requirements.txt    // [3] Cài đặt thư viện cần thiết


```

Sau lần đầu cài đặt, những lần sau chỉ cần kích hoạt [2] lại hoặc IDE sẽ tự kích hoạt thay người dùng.
