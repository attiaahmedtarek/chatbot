from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "facebook/blenderbot-400M-distill"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Chatbot is ready. Type 'quit' to exit.\n")

conversation_history = []

while True:
    input_text = input("You: ")

    if input_text.lower() in ["quit", "exit"]:
        print("Goodbye!")
        break

    conversation_history.append("User: " + input_text)

    # BlenderBot usually works best with the latest message
    inputs = tokenizer([input_text], return_tensors="pt")

    outputs = model.generate(**inputs, max_new_tokens=60)

    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    print("Bot:", response)

    conversation_history.append("Bot: " + response)
