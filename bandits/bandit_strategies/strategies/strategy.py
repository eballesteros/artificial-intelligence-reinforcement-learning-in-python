from abc import abstractmethod, abstractstaticmethod, ABC

class Strategy(ABC):
    '''
    Define strategy interface
    '''
    @abstractmethod
    def __str__(self, *args, **kargs):
        pass

    @abstractmethod
    def choose(self, *args, **kargs):
        pass
    
    @abstractmethod
    def init_bandit(self, *args, **kargs):
        pass