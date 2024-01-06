import pandas as pd
from openai import OpenAI

client = OpenAI()

prompts = pd.read_csv("./data/train_prompts.csv")

essays = pd.DataFrame(columns=["id","prompt_id","text"])

response = client.chat.completions.create(
            model="ada",
            frequency_penalty=0.5,
            messages=[
                {   
                    "role":"system",
                    "content":"Follow instructions given by user to write an essay. Try to use only the information given by user."
                },
                {   
                    "role":"user",
                    "content":"hi"
                },
            ],
            max_tokens=1000
        )
print(response)
