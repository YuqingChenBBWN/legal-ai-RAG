# LAWS90286 Template

## Setup

Setup your repository by following these steps. In the terminal:

> conda create -n . python=3.12

> conda init

--> close and reopen terminal

> conda activate .

> conda install -c conda-forge sqlite

> pip3 install -r requirements.txt

--> run `streamlit run Home.py` to confirm everything is working

> create .env file and add OPENAI_API_KEY="" with your API key

If you can't see the following already:

> create a .streamlit folder and inside this folder create a file named config.toml

> inside of the config.toml file add the following text:

```
[server]
enableXsrfProtection = false
enableCORS = false
```

## Save Your Code

> git add -A

> git commit -m "update"

> git push

Once this is done you can delete the codespaces to free up new space.

## Reopen Terminal

If you don't see the (/opt/conda/envs), use:

`conda activate .`


URL：legal-ai-rag-xxcsdjd4dwl2zsappoxuiaf.streamlit.app