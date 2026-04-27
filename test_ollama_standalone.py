import ollama
import time

client = ollama.Client(host="http://localhost:11434")
prompt = "Return 1 valid Python math formula using x and y."
print("Sending request to Ollama...")
start = time.time()
try:
    response = client.generate(model="functiongemma", prompt=prompt)
    print(f"Response in {time.time() - start:.2f}s:")
    print(response['response'])
except Exception as e:
    print(f"Error: {e}")
