from langchain.chains.base import Chain
from langchain.chains.summarize import SummarizeChain
from langchain.errors import ChainLoadError

def load_chain(name: str) -> Chain:
    if name == "summarize_chain":
        return SummarizeChain()
    else:
        raise ChainLoadError(f"Unknown chain name: {name}")