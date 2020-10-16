class Cachorro():
    def __init__(self, nome, idade):
        self.nome = nome
        self.idade = idade
    
    def sentar(self):
        print(self.nome.title() + ' está sentado.')

    def rolar(self):
        print(self.nome.title() + ' esta rolando.') 

    def anos(self):
        print(self.nome.title() + ' tem ' + str(self.idade) + ' anos de idade.')

cachorro = Cachorro('rex', 6)
cachorro.sentar()
cachorro.anos()

# Atributos privados com __
class ContaBancaria():
    def __init__(self, codigo, saldo = 0):
        self.codigo = codigo
        self.__saldo = saldo
    def deposita(self, quantia):
        pass
    def retira(self, quantia):
        pass
    def getSaldo(self):
        return self.__saldo

conta = ContaBancaria(1, 204)
print(conta.getSaldo())

# testes de unidade com unittest
import unittest
from matematica import fatorial

class MatematicaTestCase(unittest.TestCase):
    def test_fatorial(self):
        self.assertEqual(fatorial(0), 1)
        self.assertEqual(fatorial(1), 1)
        self.assertEqual(fatorial(2), 2)
        self.assertEqual(fatorial(3), 6)
    
unittest.main()
