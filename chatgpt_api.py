import openai

def query_chatgpt(prompt, context):
    """
    Combines context and query into a prompt, then calls the ChatGPT API.
    Adjust the prompt formatting as needed.
    """
    full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Change model if needed.
        messages=[{"role": "user", "content": full_prompt}]
    )
    return response.choices[0].message['content']