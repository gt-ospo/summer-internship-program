# Granite Tutorial Supplement (BETA)

This page covers some added resources to run and work with the open source [IBM Granite Workshop material](https://ibm.github.io/granite-workshop).

> [!IMPORTANT]
> You will need access to the GA Tech VPN - this can be installed locally, or run through the browser here: [vpn.gatech.edu](https://vpn.gatech.edu/)

___

## Running on PACE ICE
Students with the VSIP program have access to the ICE cluster, which has CPU and GPU backends as well as an ollama instance supported through Open OnDemand. This Ollama instance can be used in place of running locally on your laptop or Replicate, which uses a GPU backend.

[Using Ollama + Jupyter in Open OnDemand](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0044442) (requires GT login)

> [!NOTE] 
> PACE supports ollama endpoints via Slurm scheduled jobs rather than as a locally (laptop) or remotely hosted (Replicate) service. You can use the Open OnDemand (OOD) application on PACE ICE to schedule a Slurm job that then has access to a set of shared models, including the latest Granite 3 models.

## Part 1: Setting Up Your PACE Virtual Environment
We highly suggests that you create a "clean" venv or virtual environment to run the workshop notebooks. Once you log in to PACE ICE, you can run the following from a terminal on the login node:

- Step 1. Load the (stable) Python version
```
[login-ice ~]$ module load python/3.12
```

- Step 2. Declare the virtual environment
```
[login-ice ~]$ python -m venv --upgrade-deps --clear venv granite_env
```
> [!NOTE]
> We've named our venv "granite_venv" here but you can name it whatever you like!


- Step 3. Source the newly decalred environment to use it on your terminal
```
[login-ice ~]$ source ~/granite_env/bin/activate
```
> [!NOTE]
> Make sure this venv has jupyter installed to use with the PACE OOD environment.

- Step 4. Install the required python-pip libraries
```
(granite_env) [login-ice ~]$ python -m pip install --require-virtualenv notebook jupyter ipywidgets
```

## Part 2: Clone The Granite Workshop Repo
While we're still in the terminal, we need to clone the Granite workshop repo into a desired location on PACE.

- Step 1. Clone The IBM Granite Workshop Repo
```
[login-ice ~]$ git clone https://github.com/IBM/granite-workshop.git
```
- Step 2. Now when you start your Jupyter session, you'll then use the Jupyter file browser to navigate to `granite-workshop/notebooks`.


## Part 3: Starting an Open OnDemand instance
The easiest way to use Ollama with Jupyter notebooks is to use the [Open OnDemand instance for ICE](https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042133) (Requires GT login for docs and OOD interface). Once you log in to the OnDemand instance [**(Direct Access Link, GATECH VPN/LOGIN Required)**](https://ondemand-ice.pace.gatech.edu/pun/sys/dashboard/batch_connect/sys/bc_ollama_jupyter_rh9/session_contexts/new), you can select the following: `Interactive Applications->Ollama + Jupyter (Beta)`

Now you need to create an instance, for your convience - we recommend running with the following parameters:
- Python environment: Custom virtual environment
    - Virtual environment path: Wherever you created your virtual environment (see above).  In our example, we used `/home/hice1/<gburdell>/granite_venv`
- Ollama models directory: `PACE Shared Models`
- Jupyter interface: Whichever you prefer
- Node type: <Choose 1 of "NVIDIA L40s", "NVIDIA GPU H100 (HGX)", or "NVIDIA GPU (first avail)">
- Number of CPUs (cores): `4`
- Number of GPUs: `1`
- Total memory: <32 GB or 64 GB>
- Time: <start with 1 hour and expand as needed>
- Quality of service: `Default (none)`

---

## Part 4: ICE Specific Changes
Through testing, we've discovered a change that must be made in regards to how the notebooks starts an Ollama session. Our approach on ICE is actually simpler than the original notebook.

In the original [Summarize example notebook](https://ibm.github.io/granite-workshop/lab-1/) from Lab 1, the session is started in this cell. 
 
```Python
import os
import requests
from langchain_ollama.llms import OllamaLLM
from langchain_community.llms import Replicate
from ibm_granite_community.notebook_utils import get_env_var

model_path = "ibm-granite/granite-3.3-8b-instruct"
try: # Look for a locally accessible Ollama server for the model
    response = requests.get(os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
    model = OllamaLLM(
        model="granite3.3:8b",
        num_ctx=65536, # 64K context window
    )
    model = model.bind(raw=True) # Client side controls prompt
except Exception: # Use Replicate for the model
    model = Replicate(
        model=model_path,
        replicate_api_token=get_env_var('REPLICATE_API_TOKEN'),
        model_kwargs={
            "max_tokens": 2000, # Set the maximum number of tokens to generate as output.
            "min_tokens": 200, # Set the minimum number of tokens to generate as output.
            "temperature": 0.75,
            "presence_penalty": 0,
            "frequency_penalty": 0,
        },
    )
```

**However,** you must replace this cell with the code below in order to proceed with the rest of the notebook.

```Python
import os
from langchain_ollama.llms import OllamaLLM

model_path = "ibm-granite/granite-3.3-8b-instruct"
model = OllamaLLM(
    model="granite3.3:8b",
    num_ctx=65536, # 64K context window
)
model = model.bind(raw=True) # Client side controls prompt
```

> [!TIP] 
> Once you've made the change, you can print out all the available models for your Ollama instance by running the following in a new cell:
> ```
> import ollama
> for m in ollama.list().models:
>     print(m.model)
> ```

#### You're all set! Now you can rock and roll with Ollama via your browser 🚀🤘

___ 

## Frequently Asked Questions
### Can I run these locally
IBM recommends running the Granite workshop on a computer with at least 32 GB of RAM and with a GPU. Users who wish to run the workshop locally but do not have a computer that meets these requirements, may experience generation times of 10-20 minutes to run a single query on a computer with the Granite demos - along side increased heat production and resource consumption.

We encourage you to play around with these models, after all - they're open source! But please note that not everything you want to build may be possible to run on your current computer without the help of the PACE-ICE cluser.

### Possible local errors
> [!CAUTION]
> Most errors were experienced on MacOS, users are advised to use Windows or Linux if possible.

`Python version does not match the expected Python 3.10-3.13`

If you get an error when running the Granite notebooks about the Python version, you can try the following:

- Step 1. Create a Python environment with an explicit version of Python. Here we name it "granite_venv"
```bash
python3.10 -m venv --upgrade-deps --clear venv granite_venv
source granite_venv/bin/activate
python -m pip install --require-virtualenv notebook ipywidgets
```
> [!NOTE]
> You now need to start Jupyter from this venv to use this version of Python

- Step 2. Make sure you don't have other copies of Jupyter notebook or lab installed that use other versions of Python

```bash
brew list | grep jupyter
brew remove jupyter
```

> [!IMPORTANT]
> If you experience an error that was not described above, please reach out to your faculty mentor for help via relay.
