import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

def gpt4o_mini_inference(system_prompt, instruction_prompt):

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction_prompt}
        ]
    )

    inference = completion.choices[0].message.content

    return inference

def gpt4o_mini_inference_yes_no(system_prompt, instruction_prompt):

    yes_token = 6763
    no_token = 1750

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction_prompt}
        ],
        logit_bias={
            yes_token: 100,
            no_token: 100
        },
        max_tokens=1
    )

    inference = completion.choices[0].message.content

    return inference