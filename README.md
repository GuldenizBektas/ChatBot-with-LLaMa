# Chat-LLaMa
With Chat-LLaMa you can get answers about LLMs. Please make sure you check out the knowledge base. You can find knowledge base in <a href="https://github.com/GuldenizBektas/ChatBot-with-LLaMa/tree/main/data">**data**</a> folder.

Install the model from <a href="https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q2_K.bin">HuggingFace</a>.

Before running the app you need to add knowledge to a vector space. Do this by:
```
python store_vector_space.py
```
If you'd like to add a new paper add it's id to the list in `collect_data.py` file.

```
paper_ids = ['2308.10620', '2307.06435', '2303.18223', '2307.10700', '2310.11207', '2305.11828']
```

After do that add this papers inside the vector store `Faiss` by Meta.

You can run this app with this command:
```
streamlit run app.py
```

