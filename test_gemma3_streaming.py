import ollama
import time

client = ollama.Client(host="http://localhost:11434")
prompt = "Return 1 valid Python math formula using x and y."
print("Sending request to Ollama (gemma3:1b streaming)...")
start = time.time()
try:
    for chunk in client.generate(model="gemma3:1b", prompt=prompt, stream=True):
        print(chunk['response'], end="", flush=True)
    print(f"\nFinished in {time.time() - start:.2f}s")
except Exception as e:
    print(f"\nError: {e}")
