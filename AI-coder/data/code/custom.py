class MyChain(Chain):
    def __init__(self, prompt, llm):
        self.prompt = prompt
        self.llm = llm

    def run(self, input):
        return self.llm(self.prompt.format(input=input))
