class Probs():
    def __init__(self, paras):

        # loading prompt of problem
        if not isinstance(paras.problem, str):
            self.prob = paras.problem
            print("- Prob local loaded ")
        else:
            print("problem " + paras.problem + " not found!")

    def get_problem(self):

        return self.prob
