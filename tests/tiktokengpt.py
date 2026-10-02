import tiktoken

enc = tiktoken.get_encoding("gpt2")
text = "My name is Saeedullah"
tokens = enc.encode(text)
print(tokens)
for token in tokens:
    print(f"{token} -> {enc.decode_single_token_bytes(token)}")
print(enc.decode(tokens))