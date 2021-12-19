import math

class BaseEpsilon:
    """
    This is the parent class for all different epsilon schedules.
    Use ConstantEpsilon instead of this one for a constant schedule.
    """
    def __init__(self, epsilon):
        self.epsilon = epsilon
    
    def __call__(self):
        # Allows you to get the value of epsilon using epsilon()
        return self.epsilon

    def __str__(self):
        # Used for parsing to JSON, etc.
        name = self.__class__.__name__
        return f"{name}({self.__dict__})"

    def step(self, episode):
        raise NotImplementedError

class ConstantEpsilon(BaseEpsilon):
    """
    Class used for a constant epsilon schedule.
    Applying self.step has no effect on epsilon.
    """
    def __init__(self, epsilon):
        super().__init__(epsilon)

    def step(self, episode):
        pass

class LinearEpsilon(BaseEpsilon):
    """
    Class used for a linearly decreasing epsilon schedule.
    Decrements by the amount equal to 1/rate every episode.
    
    Initialises epsilon to epsilon_start and never goes below epsilon_end
    """
    def __init__(self, epsilon_start, epsilon_end, epsilon_rate):
        super().__init__(epsilon_start)

        self.start = epsilon_start
        self.end = epsilon_end
        self.rate = epsilon_rate

    def step(self, episode):
        decrement = (self.start - self.end) * max(0, (1 - episode / self.rate))
        self.epsilon = self.end + decrement
        
class DecayingEpsilon(BaseEpsilon):
    """
    Class used for an exponentially decaying epsilon schedule.

    Reduces the difference between the current and end epsilon 
    by dividing by e^(1/rate) every episode.

    Initialises epsilon to epsilon_start and eventually settles
    to epsilon end.
    """
    def __init__(self, epsilon_start, epsilon_end, epsilon_decay):
        super().__init__(epsilon_start)

        self.start = epsilon_start
        self.end = epsilon_end
        self.decay = epsilon_decay

    def step(self, episode):
        self.epsilon = self.end + \
            (self.start - self.end) * math.exp(-episode / self.decay)
