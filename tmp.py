import torch

class A():
  def __init__(self):
    self.list = torch.zeros(4)
    self.idx = torch.zeros((1)).int()
    
    self.l = []

    # self.list.share_memory_()
    # self.idx.share_memory_()

    self.B = B(self)

  def func(self):
    self.list[self.idx[0]] += 1
    self.idx[0] += 1
    self.l.append('a')

  def p(self):
    print('A')
    print('list:', self.list)
    print('idx:', self.idx)
    print('l:', self.l)
    self.B.p()
    
    self.B.func()
    self.B.p()

    print('A')
    print('list:', self.list)
    print('idx:', self.idx)
    print('l:', self.l)
    self.B.p()

    self.func()


    self.list[3] += 100
    self.l.append('A')
    print('A')
    print('list:', self.list)
    print('idx:', self.idx)
    print('l:', self.l)
    self.B.p()


    print('-'*10)



  
class B():
  def __init__(self, A):
    self.list = A.list
    self.idx = A.idx
    self.l = A.l

  def func(self):
    self.list[self.idx[0]] += 1
    self.idx[0] += 1
    self.l.append('b')

  def p(self):
    print('B')
    print('list:', self.list)
    print('idx:', self.idx)
    print('l:', self.l)

a = A()

a.p()