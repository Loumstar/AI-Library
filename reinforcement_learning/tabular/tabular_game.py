class BaseTabularEnvironment():
    def __init__(self, shape=(10, 10), obstaces=[], absorbing=[(10, 10)], rewards=[500], 
                    step_reward=-1, discount_rate=1, prob_success=0.75):
        
        pass

    def is_absorbing(self, state):
        NotImplementedError
